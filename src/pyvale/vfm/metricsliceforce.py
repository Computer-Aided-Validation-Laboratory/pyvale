from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.metric import IMetric
from pyvale.vfm.spatialparam import ISpatialParameterisation
from pyvale.vfm.spatialparamslicewise import SliceCrossSection, SlicePartition


SliceForceReconstructionMethod = Literal["weighted_line_average", "weighted_line_average_extrapolated"]
_GEOMETRY_TOLERANCE = 1.0e-9


@dataclass(slots=True)
class SliceForceReconstructionDiagnostics:
    reconstructed_force: npt.NDArray[np.float64]
    force_reconstruction_error: npt.NDArray[np.float64]
    normalised_force_reconstruction_error: npt.NDArray[np.float64]
    temporal_weights: npt.NDArray[np.float64]
    spatial_weights: npt.NDArray[np.float64]
    weighted_temporal_rms: npt.NDArray[np.float64]
    weighted_spatiotemporal_rms: float
    applied_longitudinal_force: npt.NDArray[np.float64]


@dataclass(slots=True, frozen=True)
class LocalSliceData:
    """Precomputed local force-reconstruction data for one slice."""

    slice_index: int
    axis: str
    stress_component_index: int
    global_point_indices: npt.NDArray[np.int64]
    local_point_indices: npt.NDArray[np.int64]
    point_area_integral_weights: npt.NDArray[np.float64]
    applied_longitudinal_force: npt.NDArray[np.float64]
    temporal_weights: npt.NDArray[np.float64]
    spatial_weight: float


@dataclass(slots=True)
class SliceWiseForceReconstructionMetric(IMetric):
    """Weighted slice force reconstruction metric.

    Notes
    -----
    The slice integral is computed from the weighted average of precomputed
    cross-sectional line integrals rather than by directly summing all stress
    points that fall inside the slice area.
    """

    slice_partition: SlicePartition
    method: SliceForceReconstructionMethod = "weighted_line_average"
    edge_extrapolation_points: int = 5
    edge_extrapolation_degree: int = 3
    _slice_force_point_indices: tuple[npt.NDArray[np.int64], ...] = field(init=False, repr=False)
    _slice_force_point_area_integral_weights: tuple[npt.NDArray[np.float64], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        slice_force_operators = tuple(
            _build_slice_force_operator(
                cross_sections=cross_sections,
                slice_span=float(self.slice_partition.spans[slice_index]),
                method=self.method,
                edge_extrapolation_points=self.edge_extrapolation_points,
                edge_extrapolation_degree=self.edge_extrapolation_degree,
            )
            for slice_index, cross_sections in enumerate(self.slice_partition.cross_sections)
        )
        self._slice_force_point_indices = tuple(operator[0] for operator in slice_force_operators)
        self._slice_force_point_area_integral_weights = tuple(operator[1] for operator in slice_force_operators)

    def evaluate(
        self,
        stress: npt.NDArray[np.float64],
        constitutive_law: IConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, ISpatialParameterisation],
        experiment_data: ExperimentData,
    ) -> npt.NDArray[np.float64]:
        diagnostics = self.evaluate_diagnostics(stress, experiment_data)
        weighted_residual = (
            diagnostics.normalised_force_reconstruction_error
            * np.sqrt(diagnostics.temporal_weights)[:, np.newaxis]
            * np.sqrt(diagnostics.spatial_weights)[np.newaxis, :]
        )
        return weighted_residual.ravel()

    def build_local_slice_data(
        self,
        experiment_data: ExperimentData,
        slice_index: int,
    ) -> LocalSliceData:
        """Build the local linear operator required for one independent slice solve."""

        if slice_index < 0 or slice_index >= self.slice_partition.num_slices:
            raise IndexError(f"Slice index {slice_index} is out of range.")

        applied_longitudinal_force = _extract_force_component(experiment_data.boundary_conditions.force, self.slice_partition.axis)
        temporal_weights = _normalise_weights(np.abs(applied_longitudinal_force))
        spatial_weights = _normalise_weights(self.slice_partition.widths)

        point_indices = self._slice_force_point_indices[slice_index]
        point_area_integral_weights = self._slice_force_point_area_integral_weights[slice_index]
        finite_strain_points = np.all(np.isfinite(experiment_data.strain), axis=(0, 1)).ravel()
        if point_indices.size > 0:
            valid_operator_points = finite_strain_points[point_indices]
            point_indices = point_indices[valid_operator_points]
            point_area_integral_weights = point_area_integral_weights[valid_operator_points]

        if point_indices.size == 0:
            global_point_indices = np.zeros(0, dtype=np.int64)
            local_point_indices = np.zeros(0, dtype=np.int64)
        else:
            global_point_indices, local_point_indices = np.unique(point_indices, return_inverse=True)
            global_point_indices = global_point_indices.astype(np.int64, copy=False)
            local_point_indices = local_point_indices.astype(np.int64, copy=False)

        return LocalSliceData(
            slice_index=slice_index,
            axis=self.slice_partition.axis,
            stress_component_index=0 if self.slice_partition.axis == "x" else 1,
            global_point_indices=global_point_indices,
            local_point_indices=local_point_indices,
            point_area_integral_weights=point_area_integral_weights,
            applied_longitudinal_force=applied_longitudinal_force,
            temporal_weights=temporal_weights,
            spatial_weight=float(spatial_weights[slice_index]),
        )

    def evaluate_local(
        self,
        stress: npt.NDArray[np.float64],
        experiment_data: ExperimentData,
        local_problem: LocalSliceData,
    ) -> npt.NDArray[np.float64]:
        """Evaluate the weighted residual vector for one locally solved slice."""

        if stress.ndim != 4:
            raise ValueError(f"Expected stress with shape (timesteps, components, y, x), got {stress.shape}.")
        if stress.shape[0] != local_problem.applied_longitudinal_force.shape[0]:
            raise ValueError(
                "Local stress history does not match the force history length: "
                f"{stress.shape[0]} vs {local_problem.applied_longitudinal_force.shape[0]}."
            )

        stress_component = stress[:, local_problem.stress_component_index, :, :].reshape(stress.shape[0], -1)
        reconstructed_force = np.zeros(local_problem.applied_longitudinal_force.shape, dtype=np.float64)
        if local_problem.local_point_indices.size > 0 and local_problem.point_area_integral_weights.size > 0:
            reconstructed_force = (
                float(experiment_data.specimen_geometry.thickness)
                * np.sum(
                    stress_component[:, local_problem.local_point_indices]
                    * local_problem.point_area_integral_weights[np.newaxis, :],
                    axis=1,
                )
            )

        force_error = reconstructed_force - local_problem.applied_longitudinal_force
        force_scale = float(np.max(np.abs(local_problem.applied_longitudinal_force)))
        if force_scale <= 0.0:
            force_scale = 1.0

        weighted_residual = (
            (force_error / force_scale)
            * np.sqrt(local_problem.temporal_weights)
            * np.sqrt(local_problem.spatial_weight)
        )
        return weighted_residual

    def evaluate_diagnostics(
        self,
        stress: npt.NDArray[np.float64],
        experiment_data: ExperimentData,
    ) -> SliceForceReconstructionDiagnostics:
        if stress.ndim != 4:
            raise ValueError(f"Expected stress with shape (timesteps, components, y, x), got {stress.shape}.")

        loading_axis = self.slice_partition.axis
        stress_component_index = 0 if loading_axis == "x" else 1
        applied_longitudinal_force = _extract_force_component(experiment_data.boundary_conditions.force, loading_axis)
        if applied_longitudinal_force.shape[0] != stress.shape[0]:
            raise ValueError(
                f"Force history length {applied_longitudinal_force.shape[0]} does not match stress timesteps {stress.shape[0]}."
            )

        reconstructed_force = np.zeros((stress.shape[0], self.slice_partition.num_slices), dtype=np.float64)
        stress_component = stress[:, stress_component_index, :, :].reshape(stress.shape[0], -1)
        thickness = float(experiment_data.specimen_geometry.thickness)

        for slice_index, (point_indices, point_area_integral_weights) in enumerate(
            zip(self._slice_force_point_indices, self._slice_force_point_area_integral_weights, strict=True)
        ):
            if point_indices.size == 0 or point_area_integral_weights.size == 0:
                continue
            reconstructed_force[:, slice_index] = (
                thickness * np.sum(stress_component[:, point_indices] * point_area_integral_weights[np.newaxis, :], axis=1)
            )

        force_reconstruction_error = reconstructed_force - applied_longitudinal_force[:, np.newaxis]
        force_scale = float(np.max(np.abs(applied_longitudinal_force)))
        if force_scale <= 0.0:
            force_scale = 1.0
        normalised_force_reconstruction_error = force_reconstruction_error / force_scale

        temporal_weights = _normalise_weights(np.abs(applied_longitudinal_force))
        spatial_weights = _normalise_weights(self.slice_partition.widths)

        weighted_temporal_rms = np.sqrt(
            np.sum(temporal_weights[:, np.newaxis] * normalised_force_reconstruction_error**2, axis=0)
        )
        weighted_spatiotemporal_rms = float(
            np.sqrt(
                np.sum(
                    temporal_weights[:, np.newaxis]
                    * spatial_weights[np.newaxis, :]
                    * normalised_force_reconstruction_error**2
                )
            )
        )

        return SliceForceReconstructionDiagnostics(
            reconstructed_force=reconstructed_force,
            force_reconstruction_error=force_reconstruction_error,
            normalised_force_reconstruction_error=normalised_force_reconstruction_error,
            temporal_weights=temporal_weights,
            spatial_weights=spatial_weights,
            weighted_temporal_rms=weighted_temporal_rms,
            weighted_spatiotemporal_rms=weighted_spatiotemporal_rms,
            applied_longitudinal_force=applied_longitudinal_force,
        )


def _build_slice_force_operator(
    *,
    cross_sections: tuple[SliceCrossSection, ...],
    slice_span: float,
    method: SliceForceReconstructionMethod,
    edge_extrapolation_points: int,
    edge_extrapolation_degree: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Build one linear operator that maps stress values to a slice force."""

    if slice_span <= _GEOMETRY_TOLERANCE or len(cross_sections) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    operator_indices: list[int] = []
    operator_weights: list[float] = []
    for cross_section in cross_sections:
        line_weights = np.array(cross_section.point_area_integral_weights, dtype=np.float64, copy=True)
        if method == "weighted_line_average_extrapolated":
            line_weights += _build_cross_section_extrapolation_weights(
                cross_section=cross_section,
                edge_extrapolation_points=edge_extrapolation_points,
                edge_extrapolation_degree=edge_extrapolation_degree,
            )

        scale = cross_section.width_weight / slice_span
        operator_indices.extend(cross_section.point_indices.tolist())
        operator_weights.extend((scale * line_weights).tolist())

    return np.asarray(operator_indices, dtype=np.int64), np.asarray(operator_weights, dtype=np.float64)


def _build_cross_section_extrapolation_weights(
    *,
    cross_section: SliceCrossSection,
    edge_extrapolation_points: int,
    edge_extrapolation_degree: int,
) -> npt.NDArray[np.float64]:
    """Build linear edge-correction weights for one cross-section."""

    extrapolation_weights = np.zeros(cross_section.point_area_integral_weights.shape, dtype=np.float64)
    if cross_section.point_indices.size == 0:
        return extrapolation_weights

    unique_segment_ids = np.unique(cross_section.point_segment_ids)
    for segment_id in unique_segment_ids:
        segment_mask = cross_section.point_segment_ids == segment_id
        segment_indices = np.flatnonzero(segment_mask)
        if segment_indices.size == 0:
            continue

        segment_min, segment_max = cross_section.segment_bounds[int(segment_id)]
        segment_point_coordinates = cross_section.point_coordinates[segment_mask]
        segment_point_cells = cross_section.point_cell_bounds[segment_mask]

        left_gap_min = segment_min
        left_gap_max = float(segment_point_cells[0, 0])
        if left_gap_max - left_gap_min > _GEOMETRY_TOLERANCE:
            num_fit_points = min(edge_extrapolation_points, segment_indices.size)
            local_weights = _integrated_polynomial_weights(
                sample_coordinates=segment_point_coordinates[:num_fit_points],
                integration_min=left_gap_min,
                integration_max=left_gap_max,
                polynomial_degree=edge_extrapolation_degree,
            )
            extrapolation_weights[segment_indices[:num_fit_points]] += local_weights

        right_gap_min = float(segment_point_cells[-1, 1])
        right_gap_max = segment_max
        if right_gap_max - right_gap_min > _GEOMETRY_TOLERANCE:
            num_fit_points = min(edge_extrapolation_points, segment_indices.size)
            local_weights = _integrated_polynomial_weights(
                sample_coordinates=segment_point_coordinates[-num_fit_points:],
                integration_min=right_gap_min,
                integration_max=right_gap_max,
                polynomial_degree=edge_extrapolation_degree,
            )
            extrapolation_weights[segment_indices[-num_fit_points:]] += local_weights

    return extrapolation_weights


def _integrated_polynomial_weights(
    *,
    sample_coordinates: npt.NDArray[np.float64],
    integration_min: float,
    integration_max: float,
    polynomial_degree: int,
) -> npt.NDArray[np.float64]:
    """Return the linear weights that integrate a fitted polynomial over an interval."""

    if integration_max - integration_min <= _GEOMETRY_TOLERANCE:
        return np.zeros(sample_coordinates.shape, dtype=np.float64)
    if sample_coordinates.size == 1:
        return np.asarray([integration_max - integration_min], dtype=np.float64)

    fit_degree = min(polynomial_degree, sample_coordinates.size - 1)
    integral_weights = np.zeros(sample_coordinates.size, dtype=np.float64)
    for basis_index in range(sample_coordinates.size):
        basis_values = np.zeros(sample_coordinates.size, dtype=np.float64)
        basis_values[basis_index] = 1.0
        polynomial_coefficients = np.polyfit(sample_coordinates, basis_values, fit_degree)
        integral_coefficients = np.polyint(polynomial_coefficients)
        integral_weights[basis_index] = float(
            np.polyval(integral_coefficients, integration_max)
            - np.polyval(integral_coefficients, integration_min)
        )
    return integral_weights


def _extract_force_component(
    force: npt.NDArray[np.float64],
    axis: str,
) -> npt.NDArray[np.float64]:
    resolved_force = np.asarray(force, dtype=np.float64)
    if resolved_force.ndim == 1:
        return resolved_force
    if resolved_force.ndim != 2:
        raise ValueError(f"Unsupported force array shape {resolved_force.shape}.")

    component_index = 0 if axis == "x" else 1
    if resolved_force.shape[1] <= component_index:
        raise ValueError(
            f"Force array shape {resolved_force.shape} does not contain the required {axis}-component."
        )
    return resolved_force[:, component_index]


def _normalise_weights(raw_weights: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    resolved = np.asarray(raw_weights, dtype=np.float64)
    resolved = np.where(np.isfinite(resolved) & (resolved > 0.0), resolved, 0.0)
    total = float(np.sum(resolved))
    if total <= 0.0:
        return np.full(resolved.shape, 1.0 / resolved.size, dtype=np.float64)
    return resolved / total
