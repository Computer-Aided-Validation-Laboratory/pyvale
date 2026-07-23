from __future__ import annotations

import copy
import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.metric import IMetric
from pyvale.vfm.metricsliceforce import (
    LocalSliceData,
    SliceWiseForceReconstructionMetric,
    _extract_force_component,
    _normalise_weights,
)
from pyvale.vfm.normalisation import (
    denormalise_degrees_of_freedom,
    normalise_degrees_of_freedom,
)
from pyvale.vfm.objectivefunc import IObjectiveFunction
from pyvale.vfm.optimiser import IOptimiser
from pyvale.vfm.spatialparam import ISpatialParameterisation
from pyvale.vfm.spatialparamslicewise import SliceWiseSpatialParameterisation

SliceMetricType = SliceWiseForceReconstructionMetric


class SliceWiseIndependentLeastSquares(IOptimiser):
    """Identify each slice independently using a local least-squares solve.

    This optimiser is intended for slice-wise identification with the
    slice force-reconstruction metric. All unknown parameters must
    therefore use `SliceWiseSpatialParameterisation` with the same slice
    partition object.
    """

    method: str = "trf"

    def optimise(
        self,
        constitutive_law: IConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, ISpatialParameterisation],
        metrics: list[IMetric],
        objective_function: IObjectiveFunction,
        experiment_data: ExperimentData,
    ) -> dict[str, ISpatialParameterisation]:

        # Validate the metric is compatible with independent slice-wise identification.
        if len(metrics) != 1 or not isinstance(metrics[0], SliceWiseForceReconstructionMetric):
            raise ValueError(
                "SliceWiseIndependentLeastSquares requires exactly one "
                "SliceWiseForceReconstructionMetric."
            )
        slice_metric = metrics[0]

        # Validate the parameterisations are compatible with independent slice-wise identification.
        _validate_slice_parameterisations(spatial_parameterisations, slice_metric)

        # Create a deep copy of the spatial parameterisations to avoid modifying the originals.
        optimised_spatial_parameterisations = {
            param_name: copy.deepcopy(sp)
            for param_name, sp in spatial_parameterisations.items()
        }

        # Solve each slice independently, updating the optimised parameterisations in place.
        # Note: could be parallelised in the future if needed, but this is not currently a bottleneck.
        for slice_index in range(slice_metric.slice_partition.num_slices):
            # Build the slice-local solve data, which includes point associations, point weights,
            # local strain, fixed parameter maps
            local_slice_data = _build_slice_solve_data(
                slice_index=slice_index,
                parameter_map_size=parameter_map_size,
                spatial_parameterisations=optimised_spatial_parameterisations,
                slice_metric=slice_metric,
                experiment_data=experiment_data,
            )
            # If there are no unknown parameters for this slice, skip the optimisation.
            if len(local_slice_data.unknown_parameter_names) == 0:
                continue

            # Build the initial guess for the unknown parameters for this slice, normalised to [0, 1].
            initial_guess = _build_initial_guess(
                optimised_spatial_parameterisations,
                slice_index,
                local_slice_data.unknown_parameter_names,
            )

            result = least_squares(
                fun = _evaluate_slice_candidate,
                x0 = initial_guess,
                bounds=(np.zeros_like(initial_guess), np.ones_like(initial_guess)),
                method=self.method,
                args=(
                    constitutive_law,
                    objective_function,
                    slice_metric,
                    experiment_data,
                    local_slice_data,
                ),
            )

            solved_values = denormalise_degrees_of_freedom(
                result.x,
                local_slice_data.lower_bounds,
                local_slice_data.upper_bounds,
            )
            _update_slice_parameterisations(
                optimised_spatial_parameterisations,
                slice_index,
                local_slice_data.unknown_parameter_names,
                solved_values,
            )

        return optimised_spatial_parameterisations



def _validate_slice_parameterisations(
    spatial_parameterisations: dict[str, ISpatialParameterisation],
    slice_metric: SliceMetricType,
) -> None:
    for param_name, spatial_parameterisation in spatial_parameterisations.items():
        if isinstance(spatial_parameterisation, SliceWiseSpatialParameterisation):
            if spatial_parameterisation.slice_partition is not slice_metric.slice_partition:
                raise ValueError(
                    "All SliceWiseSpatialParameterisation instances must share the same "
                    "slice partition used by the selected slice force metric."
                )
            continue

        if spatial_parameterisation.get_num_degrees_of_freedom() != 0:
            raise ValueError(
                "Independent slice-wise identification requires all unknown parameters "
                f"to use SliceWiseSpatialParameterisation. Parameter '{param_name}' does not."
            )


def _build_slice_solve_data(
    *,
    slice_index: int,
    parameter_map_size: npt.NDArray[np.uint32],
    spatial_parameterisations: dict[str, ISpatialParameterisation],
    slice_metric: SliceMetricType,
    experiment_data: ExperimentData,
) -> LocalSliceData:
    """Build all cached inputs needed to solve one slice independently.

    This converts the global identification setup into a compact slice-local
    problem for repeated use inside SciPy's least-squares iterations.

    The returned object contains:
    - the slice-local force reconstruction data
    - the strain history only at points used by this slice
    - the names and bounds of any unknown slice parameters
    - local parameter maps for parameters that are fixed during this slice solve
    """

    # Validate the slice index is within the valid range of slices.
    if slice_index < 0 or slice_index >= slice_metric.slice_partition.num_slices:
        raise IndexError(f"Slice index {slice_index} is out of range.")

    # Extract the applied longitudinal force for this slice 
    applied_longitudinal_force = _extract_force_component(
        experiment_data.boundary_conditions.force,
        slice_metric.slice_partition.axis,
    )

    # Compute the temporal and spatial weights (optionally used in objective function).
    # - temporal weights emphasise load steps with larger applied force
    # - spatial weights scale each slice relative to the others
    temporal_weights = _normalise_weights(np.abs(applied_longitudinal_force))
    spatial_weights = _normalise_weights(slice_metric.slice_partition.widths)

    # Extract the point indices for this slice
    point_indices = slice_metric.slice_partition.slice_force_point_indices[slice_index]

    # Get the point area integral weights for this slice
    # These weights are used to scale the contribution of each point in the objective function. 
    # They are derived from the area associated with each measurement point in the slice.
    point_area_integral_weights = (
        slice_metric.slice_partition.slice_force_point_area_integral_weights[slice_index]
    )

    # Filter out any points whose strain history is not full finite.
    finite_strain_points = np.all(np.isfinite(experiment_data.strain), axis=(0, 1)).ravel()

    PLOT_VALID_STRAIN_POINTS = False # Set to True to visualise valid strain points for debugging
    if PLOT_VALID_STRAIN_POINTS:
        import matplotlib.pyplot as plt
        valid_point_mask_2d = finite_strain_points.reshape(experiment_data.specimen_geometry.x.shape)
        plt.figure()
        plt.imshow(valid_point_mask_2d, origin="lower", cmap="gray")
        plt.title("Finite Strain Points: white=valid, black=invalid")
        plt.xlabel("x index")
        plt.ylabel("y index")
        plt.colorbar()
        plt.show()

    # Restrict the slice force operator to points with fully finite strain history.
    # `point_indices` contains global flat indices into the full DIC field.
    # `finite_strain_points` is a global boolean mask over all points.
    if point_indices.size > 0:
        valid_operator_points = finite_strain_points[point_indices]
        point_indices = point_indices[valid_operator_points]
        point_area_integral_weights = point_area_integral_weights[valid_operator_points]

    # `point_indices` may still contain repeated global point references.
    # We want:
    # - `global_point_indices`: the unique global points needed for this slice
    # - `local_point_indices`: how each operator entry maps into that compact local list
    if point_indices.size == 0:
        global_point_indices = np.zeros(0, dtype=np.int64)
        local_point_indices = np.zeros(0, dtype=np.int64)
    else:
        global_point_indices, local_point_indices = np.unique(point_indices, return_inverse=True)
        global_point_indices = global_point_indices.astype(np.int64, copy=False)
        local_point_indices = local_point_indices.astype(np.int64, copy=False)

    # Extract the strain history at the points used by this slice. 
    # This is a 4D array with shape (num_timesteps, num_components, 1, num_points).
    local_strain = _extract_local_strain(
        experiment_data.strain,
        global_point_indices,
    )

    # Collect the unknowns for this slice solve and cache any fixed local
    # parameter maps that do not change during optimisation.
    unknown_parameter_names: list[str] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    fixed_parameter_maps: dict[str, npt.NDArray[np.float64]] = {}

    for param_name, spatial_parameterisation in spatial_parameterisations.items():
        if isinstance(spatial_parameterisation, SliceWiseSpatialParameterisation):
            # Look up the value for this slice. 
            slice_value = spatial_parameterisation.values[slice_index]
            if slice_value is None:
                raise ValueError(
                    "SliceWiseSpatialParameterisation must be initialised with "
                    "update_from_constitutive_parameter before optimisation."
                )

            # If value of current slice is a DegreeOfFreedom, it is an unknown parameter to be solved.
            if isinstance(slice_value, DegreeOfFreedom):
                unknown_parameter_names.append(param_name)
                lower_bounds.append(slice_value.lower_bound)
                upper_bounds.append(slice_value.upper_bound)
            else:
                # If value of current slice is a float, it is a fixed parameter.
                fixed_parameter_maps[param_name] = np.full(
                    local_strain.shape[2:],
                    float(slice_value),
                    dtype=np.float64,
                )
            continue

        full_parameter_map = spatial_parameterisation.to_map(parameter_map_size)
        fixed_parameter_maps[param_name] = _extract_local_parameter_map(
            full_parameter_map,
            global_point_indices,
        )

    # Package slice local force reconstruction data containing information needed to
    # convert stress field into reconstructed force residual, together with the
    # local constitutive inputs and optimisation bookkeeping for this slice solve.
    return LocalSliceData(
        slice_index=slice_index,
        axis=slice_metric.slice_partition.axis,
        stress_component_index=0 if slice_metric.slice_partition.axis == "x" else 1,
        global_point_indices=global_point_indices,
        local_point_indices=local_point_indices,
        point_area_integral_weights=point_area_integral_weights,
        applied_longitudinal_force=applied_longitudinal_force,
        temporal_weights=temporal_weights,
        spatial_weights=float(spatial_weights[slice_index]),
        local_strain=local_strain,
        unknown_parameter_names=tuple(unknown_parameter_names),
        lower_bounds=np.asarray(lower_bounds, dtype=np.float64),
        upper_bounds=np.asarray(upper_bounds, dtype=np.float64),
        fixed_parameter_maps=fixed_parameter_maps,
    )


def _build_initial_guess(
    spatial_parameterisations: dict[str, ISpatialParameterisation],
    slice_index: int,
    unknown_parameter_names: tuple[str, ...],
) -> npt.NDArray[np.float64]:
    slice_degrees_of_freedom: list[DegreeOfFreedom] = []
    for param_name in unknown_parameter_names:
        spatial_parameterisation = spatial_parameterisations[param_name]
        if not isinstance(spatial_parameterisation, SliceWiseSpatialParameterisation):
            raise TypeError("Expected a SliceWiseSpatialParameterisation for independent slice solving.")

        slice_value = spatial_parameterisation.values[slice_index]
        if not isinstance(slice_value, DegreeOfFreedom):
            raise TypeError(
                "Expected a DegreeOfFreedom when building the independent slice-wise initial guess."
            )
        slice_degrees_of_freedom.append(copy.copy(slice_value))

    return normalise_degrees_of_freedom(slice_degrees_of_freedom)


def _extract_local_strain(
    strain: npt.NDArray[np.float64],
    global_point_indices: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    num_timesteps = strain.shape[0]
    num_components = strain.shape[1]
    flattened_strain = strain.reshape(num_timesteps, num_components, -1)
    local_strain = flattened_strain[:, :, global_point_indices]
    return local_strain.reshape(num_timesteps, num_components, 1, global_point_indices.size)


def _extract_local_parameter_map(
    parameter_map: npt.NDArray[np.float64],
    global_point_indices: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    local_values = parameter_map.ravel()[global_point_indices]
    return local_values.reshape(1, global_point_indices.size)


def _evaluate_slice_candidate(
    normalised_degrees_of_freedom: npt.NDArray[np.float64],
    constitutive_law: IConstitutiveLaw,
    objective_function: IObjectiveFunction,
    slice_metric: SliceMetricType,
    experiment_data: ExperimentData,
    local_slice_data: LocalSliceData,
) -> npt.NDArray[np.float64]:
    resolved_degrees_of_freedom = denormalise_degrees_of_freedom(
        normalised_degrees_of_freedom,
        local_slice_data.lower_bounds,
        local_slice_data.upper_bounds,
    )

    # Build the local parameter maps for this slice, combining fixed parameters and the current candidate unknowns.
    local_parameter_maps = {
        param_name: value.copy()
        for param_name, value in local_slice_data.fixed_parameter_maps.items()
    }
    # Add the current candidate unknown parameters to the local parameter maps.
    for param_name, param_value in zip(
        local_slice_data.unknown_parameter_names,
        resolved_degrees_of_freedom,
        strict=True,
    ):
        local_parameter_maps[param_name] = np.full(
            local_slice_data.local_strain.shape[2:],
            float(param_value),
            dtype=np.float64,
        )
    # Compute the stress field for this slice using the constitutive law and the local strain and parameter maps.
    local_stress = constitutive_law.calculate_stress(
        local_slice_data.local_strain,
        local_parameter_maps,
    )
    # Evaluate the slice force reconstruction metric for this candidate stress field, returning the residuals.
    metric_result = slice_metric.evaluate_local(
        local_stress,
        experiment_data,
        local_slice_data,
    )
    # Evaluate the objective function using the metric result, which should return a vector of residuals.
    objective_value = objective_function.evaluate([metric_result])
    if not isinstance(objective_value, np.ndarray):
        raise TypeError(
            "SliceWiseIndependentLeastSquares requires a vector objective "
            "function that returns an ndarray of residuals."
        )
    return np.asarray(objective_value, dtype=np.float64).ravel()


def _update_slice_parameterisations(
    spatial_parameterisations: dict[str, ISpatialParameterisation],
    slice_index: int,
    unknown_parameter_names: tuple[str, ...],
    solved_values: npt.NDArray[np.float64],
) -> None:
    for param_name, param_value in zip(unknown_parameter_names, solved_values, strict=True):
        spatial_parameterisation = spatial_parameterisations[param_name]
        if not isinstance(spatial_parameterisation, SliceWiseSpatialParameterisation):
            raise TypeError("Expected a SliceWiseSpatialParameterisation when updating slice results.")

        slice_value = spatial_parameterisation.values[slice_index]
        if not isinstance(slice_value, DegreeOfFreedom):
            raise TypeError(
                "Expected a DegreeOfFreedom when updating the solved independent slice values."
            )
        slice_value.value = float(param_value)
