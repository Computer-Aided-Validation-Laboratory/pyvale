from __future__ import annotations

import copy
import time

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.identificationresult import (
    OptimisationOutcome,
    SolveResult,
    snapshot_object,
)
from pyvale.vfm.metric import IMetric
from pyvale.vfm.metricsliceforce import (
    LocalSliceData,
    SliceWiseForceReconstructionMetric,
    _extract_force_component,
    _filter_operator_points,
    _normalise_weights,
)
from pyvale.vfm.normalisation import (
    denormalise_degrees_of_freedom,
    normalise_degrees_of_freedom,
)
from pyvale.vfm.objectivefunc import IObjectiveFunction, IVectorObjectiveFunction
from pyvale.vfm.optimiser import IOptimiser
from pyvale.vfm.progress import ProgressEvent, emit_progress
from pyvale.vfm.slicewise_utils import slice_partitions_are_equivalent
from pyvale.vfm.spatialparam import ISpatialParameterisation, get_num_degrees_of_freedom
from pyvale.vfm.spatialparamslicewise import SliceWiseSpatialParameterisation

SliceMetricType = SliceWiseForceReconstructionMetric


def _get_single_spatial_parameterisation(
    param_name: str,
    spatial_parameterisations: list[ISpatialParameterisation],
) -> ISpatialParameterisation:
    if len(spatial_parameterisations) != 1:
        raise ValueError(
            "SliceWiseIndependentLeastSquares currently requires exactly one spatial "
            f"parameterisation per parameter. Parameter '{param_name}' has {len(spatial_parameterisations)}."
        )
    return spatial_parameterisations[0]


class SliceWiseIndependentLeastSquares(IOptimiser):
    """Identify each slice independently using a local least-squares solve."""

    method: str = "trf"

    def get_required_objective_function_type(self) -> type:
        return IVectorObjectiveFunction

    def optimise(
        self,
        constitutive_law: IConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
        metrics: list[IMetric],
        objective_function: IObjectiveFunction,
        experiment_data: ExperimentData,
        progress_callback=None,
    ) -> OptimisationOutcome:

        if len(metrics) != 1 or not isinstance(metrics[0], SliceWiseForceReconstructionMetric):
            raise ValueError(
                "SliceWiseIndependentLeastSquares requires exactly one "
                "SliceWiseForceReconstructionMetric."
            )
        slice_metric = metrics[0]
        if slice_metric.slice_partition is None:
            raise RuntimeError("Slice metric partition must be prepared before optimisation.")

        _validate_slice_parameterisations(spatial_parameterisations, slice_metric)

        optimised_spatial_parameterisations = copy.deepcopy(
            spatial_parameterisations
        )
        parent_started_at = time.perf_counter()
        parent_initial_dofs = _collect_spatial_dof_values(
            spatial_parameterisations
        )
        child_results: list[SolveResult] = []
        skipped_slice_count = 0

        # Solve each slice indentently, updating the optimised parameterisations in place
        # Note: could be parallised in future if required
        for slice_index in range(slice_metric.slice_partition.num_slices):
            local_slice_data = _build_slice_solve_data(
                slice_index=slice_index,
                parameter_map_size=parameter_map_size,
                spatial_parameterisations=optimised_spatial_parameterisations,
                slice_metric=slice_metric,
                experiment_data=experiment_data,
            )
            # If no unknown parameters for this slice, skip the solve and leave the parameterisations unchanged
            if len(local_slice_data.unknown_parameter_names) == 0:
                skipped_slice_count += 1
                continue

            if (
                local_slice_data.local_point_indices.size == 0
                or local_slice_data.point_area_integral_weights.size == 0
            ):
                raise ValueError(
                    f"Slice {slice_index} has unknown parameters but no usable points "
                    "after filtering non-finite strain histories."
                )

            # Build the initial guess for the unknown parameters of this slice, normalised to [0, 1]
            initial_guess = _build_initial_guess(
                optimised_spatial_parameterisations,
                slice_index,
                local_slice_data.unknown_parameter_names,
            )

            _emit_slice_progress(
                progress_callback,
                kind="slice_started",
                slice_index=slice_index,
                slice_count=slice_metric.slice_partition.num_slices,
            )
            slice_started_at = time.perf_counter()
            result = least_squares(
                fun=_evaluate_slice_candidate,
                x0=initial_guess,
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
            slice_runtime = time.perf_counter() - slice_started_at
            _emit_slice_progress(
                progress_callback,
                kind="slice_finished",
                slice_index=slice_index,
                slice_count=slice_metric.slice_partition.num_slices,
                evaluation_count=int(result.nfev),
                elapsed_seconds=slice_runtime,
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
            child_results.append(
                SolveResult(
                    solve_iteration=slice_index,
                    optimiser=snapshot_object(
                        self,
                        options={"method": self.method},
                    ),
                    runtime_seconds=slice_runtime,
                    num_evaluations=int(result.nfev),
                    success=bool(result.success),
                    status=int(result.status),
                    message=str(result.message),
                    initial_dofs=[
                        float(value)
                        for value in denormalise_degrees_of_freedom(
                            initial_guess,
                            local_slice_data.lower_bounds,
                            local_slice_data.upper_bounds,
                        )
                    ],
                    final_dofs=[float(value) for value in solved_values],
                    final_objective=_summarise_least_squares_result(result),
                    details={
                        "slice_index": int(slice_index),
                        "unknown_parameter_names": [
                            str(name)
                            for name in local_slice_data.unknown_parameter_names
                        ],
                        "num_local_points": int(
                            local_slice_data.global_point_indices.size
                        ),
                    },
                )
            )

        parent_runtime = time.perf_counter() - parent_started_at
        num_evaluations = sum(
            child.num_evaluations or 0
            for child in child_results
        )
        parent_success = all(
            child.success is not False
            for child in child_results
        )
        return OptimisationOutcome(
            spatial_parameterisations=optimised_spatial_parameterisations,
            solve_result=SolveResult(
                solve_iteration=0,
                optimiser=snapshot_object(
                    self,
                    options={"method": self.method},
                ),
                runtime_seconds=parent_runtime,
                num_evaluations=int(num_evaluations),
                success=parent_success,
                status="completed" if parent_success else "completed_with_failures",
                message=(
                    "Solved each active slice independently using SciPy "
                    "least_squares."
                ),
                initial_dofs=parent_initial_dofs,
                final_dofs=_collect_spatial_dof_values(
                    optimised_spatial_parameterisations
                ),
                details={
                    "num_slices": int(slice_metric.slice_partition.num_slices),
                    "solved_slice_count": int(len(child_results)),
                    "skipped_slice_count": int(skipped_slice_count),
                },
                children=child_results,
            ),
        )


def _emit_slice_progress(
    progress_callback,
    *,
    kind: str,
    slice_index: int,
    slice_count: int,
    evaluation_count: int | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    if progress_callback is None:
        return

    readable_index = slice_index + 1
    message = f"Slice {readable_index}/{slice_count} started"
    if kind == "slice_finished":
        message = f"Slice {readable_index}/{slice_count} finished"
        if evaluation_count is not None:
            message += f", evaluations: {evaluation_count}"

    emit_progress(
        progress_callback,
        ProgressEvent(
            message,
            kind=kind,
            elapsed_seconds=elapsed_seconds,
        ),
    )


def _collect_spatial_dof_values(
    spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
) -> list[float]:
    values: list[float] = []
    for parameterisation_list in spatial_parameterisations.values():
        for parameterisation in parameterisation_list:
            values.extend(
                float(dof.value)
                for dof in parameterisation.collect_degrees_of_freedom()
            )
    return values


def _summarise_least_squares_result(
    result,
) -> dict:
    residual = np.asarray(result.fun, dtype=np.float64).ravel()
    finite_residual = residual[np.isfinite(residual)]
    residual_norm = (
        None
        if finite_residual.size == 0
        else float(np.linalg.norm(finite_residual))
    )
    return {
        "cost": float(result.cost),
        "residual_norm": residual_norm,
        "residual_size": int(residual.size),
        "finite_residual_count": int(finite_residual.size),
        "optimality": float(result.optimality),
    }


def _validate_slice_parameterisations(
    spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
    slice_metric: SliceMetricType,
) -> None:
    if slice_metric.slice_partition is None:
        raise RuntimeError("Slice metric partition must be prepared before optimisation.")

    for param_name, sps in spatial_parameterisations.items():
        spatial_parameterisation = _get_single_spatial_parameterisation(param_name, sps)
        if isinstance(spatial_parameterisation, SliceWiseSpatialParameterisation):
            if spatial_parameterisation.slice_partition is None:
                raise RuntimeError(
                    "SliceWiseSpatialParameterisation partitions must be prepared before "
                    "independent slice-wise optimisation."
                )
            if not slice_partitions_are_equivalent(
                spatial_parameterisation.slice_partition,
                slice_metric.slice_partition,
            ):
                raise ValueError(
                    "All SliceWiseSpatialParameterisation instances must resolve to the "
                    "same slice partition used by the selected slice force metric."
                )
            continue

        if get_num_degrees_of_freedom(sps) != 0:
            raise ValueError(
                "Independent slice-wise identification requires all unknown parameters "
                f"to use SliceWiseSpatialParameterisation. Parameter '{param_name}' does not."
            )


def _build_slice_solve_data(
    *,
    slice_index: int,
    parameter_map_size: npt.NDArray[np.uint32],
    spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
    slice_metric: SliceMetricType,
    experiment_data: ExperimentData,
) -> LocalSliceData:
    """ Build all cached inputs needed to solve one slice independently.

    This converts the global problem into a local problem for one slice, 
    for repeated solving.
    
    The returned object contains:
    - the local slice force reconstruction data
    - the strain history for the points in the slice (filtered to only include points with finite strain histories)
    - the applied longitudinal force history
    - the names and bounds of any unknown parameters for this slice
    - the fixed parameter maps for any known parameters
    """

    if slice_metric.slice_partition is None:
        raise RuntimeError("Slice metric partition must be prepared before optimisation.")
    if slice_index < 0 or slice_index >= slice_metric.slice_partition.num_slices:
        raise IndexError(f"Slice index {slice_index} is out of range.")

    applied_longitudinal_force = _extract_force_component(
        experiment_data.boundary_conditions.force,
        slice_metric.slice_partition.axis,
    )

    # Compute the temporal and spatial weights for this slice (optionally used in objective function)
    # - temporal weights are normalised absolute applied longitudinal force
    # - spatial weights are normalised slice widths
    temporal_weights = _normalise_weights(np.abs(applied_longitudinal_force))
    spatial_weights = _normalise_weights(slice_metric.slice_partition.widths)

    point_indices = slice_metric.slice_partition.slice_force_point_indices[slice_index]
    # Get the area integral weights for the points in this slice, which are used to weight the residuals in the objective function
    # These weights are the area integral of the points in the slice, which is used to weight the residuals in the objective function
    point_area_integral_weights = (
        slice_metric.slice_partition.slice_force_point_area_integral_weights[slice_index]
    )

    # Filter out any points that have non-finite strain histories, as these cannot be used in the optimisation
    finite_strain_points = np.all(np.isfinite(experiment_data.strain), axis=(0, 1)).ravel()
    point_indices, point_area_integral_weights = _filter_operator_points(
        point_indices,
        point_area_integral_weights,
        finite_strain_points,
    )

    PLOT_VALID_STRAIN_POINTS = False # set to True to visualise the points that have valid strain histories for this slice
    if PLOT_VALID_STRAIN_POINTS:
        import matplotlib.pyplot as plt
        valid_point_mask_2d = finite_strain_points.reshape(experiment_data.specimen_geometry.x.shape)
        plt.figure()
        plt.imshow(valid_point_mask_2d, cmap="gray", origin="lower")
        plt.title(f"Valid Strain Points for Slice {slice_index}. white = valid, black = invalid")
        plt.xlabel("X Index")
        plt.ylabel("Y Index")
        plt.colorbar(label="Valid Strain Point (1 = valid, 0 = invalid)")
        plt.show()

    if point_indices.size == 0:
        global_point_indices = np.zeros(0, dtype=np.int64)
        local_point_indices = np.zeros(0, dtype=np.int64)
    else:
        global_point_indices, local_point_indices = np.unique(point_indices, return_inverse=True)
        global_point_indices = global_point_indices.astype(np.int64, copy=False)
        local_point_indices = local_point_indices.astype(np.int64, copy=False)

    # Extract the local strain history for the points in this slice (4D array with shape [num_timesteps, num_components, 1, num_points_in_slice])
    local_strain = _extract_local_strain(
        experiment_data.strain,
        global_point_indices,
    )

    unknown_parameter_names: list[str] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    fixed_parameter_maps: dict[str, npt.NDArray[np.float64]] = {}

    for param_name, sps in spatial_parameterisations.items():
        spatial_parameterisation = _get_single_spatial_parameterisation(param_name, sps)
        if isinstance(spatial_parameterisation, SliceWiseSpatialParameterisation):
            # Look up the value for this slice
            if spatial_parameterisation.values is None:
                raise ValueError("SliceWiseSpatialParameterisation values must be initialised.")

            slice_value = spatial_parameterisation.values[slice_index]
            if slice_value is None:
                raise ValueError(
                    "SliceWiseSpatialParameterisation must be initialised with "
                    "initialise_from_constitutive_parameter before optimisation."
                )

            if isinstance(slice_value, DegreeOfFreedom):
                unknown_parameter_names.append(param_name)
                lower_bounds.append(slice_value.lower_bound)
                upper_bounds.append(slice_value.upper_bound)
            else:
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
    spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
    slice_index: int,
    unknown_parameter_names: tuple[str, ...],
) -> npt.NDArray[np.float64]:
    slice_degrees_of_freedom: list[DegreeOfFreedom] = []
    for param_name in unknown_parameter_names:
        spatial_parameterisation = _get_single_spatial_parameterisation(
            param_name,
            spatial_parameterisations[param_name],
        )
        if not isinstance(spatial_parameterisation, SliceWiseSpatialParameterisation):
            raise TypeError("Expected a SliceWiseSpatialParameterisation for independent slice solving.")
        if spatial_parameterisation.values is None:
            raise ValueError("SliceWiseSpatialParameterisation values must be initialised.")

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

    # Build local parameter maps for this slice, combining the fixed known parameters and the unknown parameters being solved for
    local_parameter_maps = {
        param_name: value.copy()
        for param_name, value in local_slice_data.fixed_parameter_maps.items()
    }

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

    # Compute stress field for this slice
    local_stress = constitutive_law.calculate_stress(
        local_slice_data.local_strain,
        local_parameter_maps,
    )

    # Evaluate the force reconstruction metric for this slice, which computes the residuals between the reconstructed and applied forces
    metric_result = slice_metric.evaluate_local(
        local_stress,
        experiment_data,
        local_slice_data,
    )

    # Evaluate the objective function for this slice, which computes the cost term to be minimised in the optimisation
    objective_value = objective_function.evaluate([metric_result])
    if not isinstance(objective_value, np.ndarray):
        raise TypeError(
            "SliceWiseIndependentLeastSquares requires a vector objective "
            "function that returns an ndarray of residuals."
        )
    return np.asarray(objective_value, dtype=np.float64).ravel()


def _update_slice_parameterisations(
    spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
    slice_index: int,
    unknown_parameter_names: tuple[str, ...],
    solved_values: npt.NDArray[np.float64],
) -> None:
    for param_name, param_value in zip(unknown_parameter_names, solved_values, strict=True):
        spatial_parameterisation = _get_single_spatial_parameterisation(
            param_name,
            spatial_parameterisations[param_name],
        )
        if not isinstance(spatial_parameterisation, SliceWiseSpatialParameterisation):
            raise TypeError("Expected a SliceWiseSpatialParameterisation when updating slice results.")
        if spatial_parameterisation.values is None:
            raise ValueError("SliceWiseSpatialParameterisation values must be initialised.")

        slice_value = spatial_parameterisation.values[slice_index]
        if not isinstance(slice_value, DegreeOfFreedom):
            raise TypeError(
                "Expected a DegreeOfFreedom when updating the solved independent slice values."
            )
        slice_value.value = float(param_value)
