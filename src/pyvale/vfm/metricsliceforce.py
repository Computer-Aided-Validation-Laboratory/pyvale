from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.metric import IMetric
from pyvale.vfm.spatialparam import ISpatialParameterisation
from pyvale.vfm.spatialparamslicewise import SliceAreaPartition


@dataclass(slots=True, frozen=True)
class LocalSliceData:
    """Cached data for evaluating and solving one slice independently.

    `point_area_integral_weights` and the point index arrays together form the
    discrete operator that maps local stress samples to reconstructed slice
    force for this slice.

    This object also stores the local constitutive inputs and optimisation
    bookkeeping so one slice can be solved repeatedly without rebuilding the
    same local data on every SciPy iteration.
    """

    slice_index: int
    axis: str
    stress_component_index: int
    global_point_indices: npt.NDArray[np.int64]
    local_point_indices: npt.NDArray[np.int64]
    point_area_integral_weights: npt.NDArray[np.float64]
    applied_longitudinal_force: npt.NDArray[np.float64]
    temporal_weights: npt.NDArray[np.float64]
    spatial_weights: float
    local_strain: npt.NDArray[np.float64]
    unknown_parameter_names: tuple[str, ...]
    lower_bounds: npt.NDArray[np.float64]
    upper_bounds: npt.NDArray[np.float64]
    fixed_parameter_maps: dict[str, npt.NDArray[np.float64]]


@dataclass(slots=True, frozen=True)
class SliceForceMetricResult:
    """Raw slice-force residual vector plus optional weighting metadata."""

    raw_residual: npt.NDArray[np.float64]
    normalised_residual: npt.NDArray[np.float64]
    temporal_weights: npt.NDArray[np.float64] | None
    spatial_weights: npt.NDArray[np.float64] | float | None
    reconstructed_force: npt.NDArray[np.float64]
    applied_longitudinal_force: npt.NDArray[np.float64]


@dataclass(slots=True, frozen=True)
class ForceReconstructionErrorResult:
    """Diagnostics for global slice-force reconstruction across all slices."""

    metric_result: SliceForceMetricResult
    weighted_temporal_rms: npt.NDArray[np.float64]
    weighted_spatiotemporal_rms: float


def build_local_slice_force_result(
    *,
    reconstructed_force: npt.NDArray[np.float64],
    applied_longitudinal_force: npt.NDArray[np.float64],
    temporal_weights: npt.NDArray[np.float64],
    spatial_weight: float,
) -> SliceForceMetricResult:
    """Build the raw local slice residual together with its weighting metadata."""

    raw_residual = reconstructed_force - applied_longitudinal_force
    force_scale = float(np.max(np.abs(applied_longitudinal_force)))
    if force_scale <= 0.0:
        force_scale = 1.0

    return SliceForceMetricResult(
        raw_residual=raw_residual,
        normalised_residual=raw_residual / force_scale,
        temporal_weights=np.asarray(temporal_weights, dtype=np.float64),
        spatial_weights=float(spatial_weight),
        reconstructed_force=np.asarray(reconstructed_force, dtype=np.float64),
        applied_longitudinal_force=np.asarray(applied_longitudinal_force, dtype=np.float64),
    )


def build_force_reconstruction_error_result(
    *,
    reconstructed_force: npt.NDArray[np.float64],
    applied_longitudinal_force: npt.NDArray[np.float64],
    temporal_weights: npt.NDArray[np.float64],
    spatial_weights: npt.NDArray[np.float64],
) -> ForceReconstructionErrorResult:
    """Build the raw global slice residual and weighted diagnostic summaries."""

    raw_residual = reconstructed_force - applied_longitudinal_force[:, np.newaxis]
    force_scale = float(np.max(np.abs(applied_longitudinal_force)))
    if force_scale <= 0.0:
        force_scale = 1.0

    normalised_residual = raw_residual / force_scale
    weighted_temporal_rms = np.sqrt(
        np.sum(np.asarray(temporal_weights, dtype=np.float64)[:, np.newaxis] * normalised_residual**2, axis=0)
    )
    weighted_spatiotemporal_rms = float(
        np.sqrt(
            np.sum(
                np.asarray(temporal_weights, dtype=np.float64)[:, np.newaxis]
                * np.asarray(spatial_weights, dtype=np.float64)[np.newaxis, :]
                * normalised_residual**2
            )
        )
    )

    return ForceReconstructionErrorResult(
        metric_result=SliceForceMetricResult(
            raw_residual=raw_residual,
            normalised_residual=normalised_residual,
            temporal_weights=np.asarray(temporal_weights, dtype=np.float64),
            spatial_weights=np.asarray(spatial_weights, dtype=np.float64),
            reconstructed_force=np.asarray(reconstructed_force, dtype=np.float64),
            applied_longitudinal_force=np.asarray(applied_longitudinal_force, dtype=np.float64),
        ),
        weighted_temporal_rms=weighted_temporal_rms,
        weighted_spatiotemporal_rms=weighted_spatiotemporal_rms,
    )


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


def _filter_operator_points(
    point_indices: npt.NDArray[np.int64],
    point_area_integral_weights: npt.NDArray[np.float64],
    valid_global_points: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Filter a slice force operator down to globally valid flat point indices."""

    if point_indices.size == 0:
        return point_indices, point_area_integral_weights

    valid_operator_points = valid_global_points[point_indices]
    return (
        point_indices[valid_operator_points],
        point_area_integral_weights[valid_operator_points],
    )


@dataclass(slots=True)
class SliceWiseAreaForceReconstructionMetric(IMetric):
    """Area-based slice force reconstruction metric.

    The reconstructed slice force is the discrete analogue of

        F_r = h / L_r * integral_{S_r} sigma dA

    where the area integral is approximated by precomputed overlap areas
    between each slice region and the native DIC support cells.
    """

    slice_partition: SliceAreaPartition

    def evaluate(
        self,
        stress: npt.NDArray[np.float64],
        constitutive_law: IConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, ISpatialParameterisation],
        experiment_data: ExperimentData,
    ) -> SliceForceMetricResult:
        return self.evaluate_force_recon_error(stress, experiment_data).metric_result

    def evaluate_local(
        self,
        stress: npt.NDArray[np.float64],
        experiment_data: ExperimentData,
        local_slice_data: LocalSliceData,
    ) -> SliceForceMetricResult:
        """Return the raw local slice residual plus weighting metadata."""

        if stress.ndim != 4:
            raise ValueError(f"Expected stress with shape (timesteps, components, y, x), got {stress.shape}.")
        if stress.shape[0] != local_slice_data.applied_longitudinal_force.shape[0]:
            raise ValueError(
                "Local stress history does not match the force history length: "
                f"{stress.shape[0]} vs {local_slice_data.applied_longitudinal_force.shape[0]}."
            )

        stress_component = stress[:, local_slice_data.stress_component_index, :, :].reshape(stress.shape[0], -1)
        reconstructed_force = np.zeros(local_slice_data.applied_longitudinal_force.shape, dtype=np.float64)
        if local_slice_data.local_point_indices.size > 0 and local_slice_data.point_area_integral_weights.size > 0:
            reconstructed_force = (
                float(experiment_data.specimen_geometry.thickness)
                * np.sum(
                    stress_component[:, local_slice_data.local_point_indices]
                    * local_slice_data.point_area_integral_weights[np.newaxis, :],
                    axis=1,
                )
            )

        return build_local_slice_force_result(
            reconstructed_force=reconstructed_force,
            applied_longitudinal_force=local_slice_data.applied_longitudinal_force,
            temporal_weights=local_slice_data.temporal_weights,
            spatial_weight=local_slice_data.spatial_weights,
        )

    def evaluate_force_recon_error(
        self,
        stress: npt.NDArray[np.float64],
        experiment_data: ExperimentData,
    ) -> ForceReconstructionErrorResult:
        if stress.ndim != 4:
            raise ValueError(f"Expected stress with shape (timesteps, components, y, x), got {stress.shape}.")

        loading_axis = self.slice_partition.axis
        stress_component_index = 0 if loading_axis == "x" else 1
        applied_longitudinal_force = _extract_force_component(experiment_data.boundary_conditions.force, loading_axis)
        if applied_longitudinal_force.shape[0] != stress.shape[0]:
            raise ValueError(
                f"Force history length {applied_longitudinal_force.shape[0]} does not match stress timesteps {stress.shape[0]}."
            )

        stress_component = stress[:, stress_component_index, :, :].reshape(stress.shape[0], -1)
        reconstructed_force = np.zeros((stress.shape[0], self.slice_partition.num_slices), dtype=np.float64)
        thickness = float(experiment_data.specimen_geometry.thickness)
        finite_strain_points = np.all(np.isfinite(experiment_data.strain), axis=(0, 1)).ravel()
        finite_stress_points = np.all(np.isfinite(stress_component), axis=0)
        valid_global_points = finite_strain_points & finite_stress_points

        for slice_index, (point_indices, point_area_integral_weights) in enumerate(
            zip(
                self.slice_partition.slice_force_point_indices,
                self.slice_partition.slice_force_point_area_integral_weights,
                strict=True,
            )
        ):
            point_indices, point_area_integral_weights = _filter_operator_points(
                point_indices,
                point_area_integral_weights,
                valid_global_points,
            )
            if point_indices.size == 0 or point_area_integral_weights.size == 0:
                continue
            reconstructed_force[:, slice_index] = (
                thickness * np.sum(stress_component[:, point_indices] * point_area_integral_weights[np.newaxis, :], axis=1)
            )

        temporal_weights = _normalise_weights(np.abs(applied_longitudinal_force))
        spatial_weights = _normalise_weights(self.slice_partition.widths)

        return build_force_reconstruction_error_result(
            reconstructed_force=reconstructed_force,
            applied_longitudinal_force=applied_longitudinal_force,
            temporal_weights=temporal_weights,
            spatial_weights=spatial_weights,
        )


SliceWiseForceReconstructionMetric = SliceWiseAreaForceReconstructionMetric
