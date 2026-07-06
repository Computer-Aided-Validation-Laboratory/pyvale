from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.metric import IMetric
from pyvale.vfm.metricsliceforce import (
    SliceLocalForceReconstructionProblem,
    SliceWiseForceReconstructionMetric,
)
from pyvale.vfm.metricsliceforcearea import SliceWiseAreaForceReconstructionMetric
from pyvale.vfm.normalisation import (
    denormalise_degrees_of_freedom,
    normalise_degrees_of_freedom,
)
from pyvale.vfm.objectivefunc import IObjectiveFunction
from pyvale.vfm.optimiser import IOptimiser
from pyvale.vfm.spatialparam import ISpatialParameterisation
from pyvale.vfm.spatialparamslicewise import SliceWiseSpatialParameterisation


@dataclass(slots=True, frozen=True)
class _SliceSolveData:
    """All precomputed data required to solve one slice independently."""

    local_problem: SliceLocalForceReconstructionProblem
    local_strain: npt.NDArray[np.float64]
    unknown_parameter_names: tuple[str, ...]
    lower_bounds: npt.NDArray[np.float64]
    upper_bounds: npt.NDArray[np.float64]
    fixed_parameter_maps: dict[str, npt.NDArray[np.float64]]


SliceMetricType = SliceWiseForceReconstructionMetric | SliceWiseAreaForceReconstructionMetric


@dataclass(slots=True)
class SliceWiseIndependentLeastSquares(IOptimiser):
    """Identify each slice independently using a local least-squares solve.

    This optimiser is intended for slice-wise identification with the
    selected slice force-reconstruction metric. All unknown parameters must
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
        slice_metric = _resolve_slice_metric(metrics)
        _validate_slice_parameterisations(spatial_parameterisations, slice_metric)

        optimised_spatial_parameterisations = {
            param_name: copy.deepcopy(sp)
            for param_name, sp in spatial_parameterisations.items()
        }

        for slice_index in range(slice_metric.slice_partition.num_slices):
            slice_solve_data = _build_slice_solve_data(
                slice_index=slice_index,
                parameter_map_size=parameter_map_size,
                spatial_parameterisations=optimised_spatial_parameterisations,
                slice_metric=slice_metric,
                experiment_data=experiment_data,
            )
            if len(slice_solve_data.unknown_parameter_names) == 0:
                continue

            initial_guess = _build_initial_guess(
                optimised_spatial_parameterisations,
                slice_index,
                slice_solve_data.unknown_parameter_names,
            )

            result = least_squares(
                _evaluate_slice_candidate,
                initial_guess,
                bounds=(np.zeros_like(initial_guess), np.ones_like(initial_guess)),
                method=self.method,
                args=(
                    constitutive_law,
                    objective_function,
                    slice_metric,
                    experiment_data,
                    slice_solve_data,
                ),
            )

            solved_values = denormalise_degrees_of_freedom(
                result.x,
                slice_solve_data.lower_bounds,
                slice_solve_data.upper_bounds,
            )
            _update_slice_parameterisations(
                optimised_spatial_parameterisations,
                slice_index,
                slice_solve_data.unknown_parameter_names,
                solved_values,
            )

        return optimised_spatial_parameterisations


def _resolve_slice_metric(metrics: list[IMetric]) -> SliceMetricType:
    if len(metrics) != 1 or not isinstance(
        metrics[0],
        (SliceWiseForceReconstructionMetric, SliceWiseAreaForceReconstructionMetric),
    ):
        raise ValueError(
            "SliceWiseIndependentLeastSquares requires exactly one "
            "SliceWiseForceReconstructionMetric or "
            "SliceWiseAreaForceReconstructionMetric."
        )
    return metrics[0]


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
) -> _SliceSolveData:
    local_problem = slice_metric.build_local_problem(experiment_data, slice_index)

    # Extract the slice support points once so each objective evaluation only
    # needs to update the local constitutive parameter values.
    local_strain = _extract_local_strain(
        experiment_data.strain,
        local_problem.global_point_indices,
    )

    unknown_parameter_names: list[str] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    fixed_parameter_maps: dict[str, npt.NDArray[np.float64]] = {}

    for param_name, spatial_parameterisation in spatial_parameterisations.items():
        if isinstance(spatial_parameterisation, SliceWiseSpatialParameterisation):
            slice_value = spatial_parameterisation.values[slice_index]
            if slice_value is None:
                raise ValueError(
                    "SliceWiseSpatialParameterisation must be initialised with "
                    "update_from_constitutive_parameter before optimisation."
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
            local_problem.global_point_indices,
        )

    return _SliceSolveData(
        local_problem=local_problem,
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
    slice_solve_data: _SliceSolveData,
) -> npt.NDArray[np.float64]:
    resolved_degrees_of_freedom = denormalise_degrees_of_freedom(
        normalised_degrees_of_freedom,
        slice_solve_data.lower_bounds,
        slice_solve_data.upper_bounds,
    )

    local_parameter_maps = {
        param_name: value.copy()
        for param_name, value in slice_solve_data.fixed_parameter_maps.items()
    }
    for param_name, param_value in zip(
        slice_solve_data.unknown_parameter_names,
        resolved_degrees_of_freedom,
        strict=True,
    ):
        local_parameter_maps[param_name] = np.full(
            slice_solve_data.local_strain.shape[2:],
            float(param_value),
            dtype=np.float64,
        )

    local_stress = constitutive_law.calculate_stress(
        slice_solve_data.local_strain,
        local_parameter_maps,
    )
    metric_result = slice_metric.evaluate_local(
        local_stress,
        experiment_data,
        slice_solve_data.local_problem,
    )
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
