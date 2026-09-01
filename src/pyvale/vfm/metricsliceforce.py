from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.experimentdata import ExperimentData, SpecimenGeometry
from pyvale.vfm.metric import IMetric, MetricResult
from pyvale.vfm.slicewise_utils import (
    SliceAreaPartition,
    SliceConfig,
    calculate_roi_slice_areas,
    resolve_slice_partition,
)
from pyvale.vfm.spatialparam import ISpatialParameterisation
from pyvale.vfm.spatialparamslicewise import SupportSlice


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
    force_integration_scale: float
    applied_longitudinal_force: npt.NDArray[np.float64]
    temporal_weights: npt.NDArray[np.float64]
    spatial_weights: float
    local_strain: npt.NDArray[np.float64]
    unknown_parameter_names: tuple[str, ...]
    lower_bounds: npt.NDArray[np.float64]
    upper_bounds: npt.NDArray[np.float64]
    fixed_parameter_maps: dict[str, npt.NDArray[np.float64]]


@dataclass(slots=True, frozen=True)
class ForceReconstructionErrorResult:
    """Diagnostics for global slice-force reconstruction across all slices."""

    metric_result: MetricResult
    weighted_temporal_rms: npt.NDArray[np.float64]
    weighted_spatiotemporal_rms: float


def _build_slice_metric_result(
    *,
    raw_residual: npt.NDArray[np.float64],
    reconstructed_force: npt.NDArray[np.float64],
    applied_longitudinal_force: npt.NDArray[np.float64],
    temporal_weights: npt.NDArray[np.float64],
    spatial_weights: npt.NDArray[np.float64] | float,
) -> MetricResult:
    # Raw residual is the difference between reconstructed and applied forces,
    resolved_raw_residual = np.asarray(raw_residual, dtype=np.float64)
    
    # Normalised residual is relative to applied force, FRE(t) = (F_R(t) - F_A(t)) / F_A(t)
    normalised_residual = _compute_relative_force_residual(
        raw_residual=resolved_raw_residual,
        applied_longitudinal_force=applied_longitudinal_force,
    )

    return MetricResult(
        residual=resolved_raw_residual,
        additional_fields={
            "raw_residual": resolved_raw_residual,
            "normalised_residual": normalised_residual,
            "temporal_weights": np.asarray(temporal_weights, dtype=np.float64),
            "spatial_weights": np.asarray(spatial_weights, dtype=np.float64),
            "reconstructed_force": np.asarray(reconstructed_force, dtype=np.float64),
            "applied_longitudinal_force": np.asarray(
                applied_longitudinal_force,
                dtype=np.float64,
            ),
        },
    )


def build_local_slice_force_result(
    *,
    reconstructed_force: npt.NDArray[np.float64],
    applied_longitudinal_force: npt.NDArray[np.float64],
    temporal_weights: npt.NDArray[np.float64],
    spatial_weight: float,
) -> MetricResult:
    """Build the local slice residual with weighting metadata."""

    raw_residual = reconstructed_force - applied_longitudinal_force
    return _build_slice_metric_result(
        raw_residual=raw_residual,
        reconstructed_force=reconstructed_force,
        applied_longitudinal_force=applied_longitudinal_force,
        temporal_weights=temporal_weights,
        spatial_weights=spatial_weight,
    )


def build_force_reconstruction_error_result(
    *,
    reconstructed_force: npt.NDArray[np.float64],
    applied_longitudinal_force: npt.NDArray[np.float64],
    temporal_weights: npt.NDArray[np.float64],
    spatial_weights: npt.NDArray[np.float64],
) -> ForceReconstructionErrorResult:
    """Build global residuals and weighted diagnostic summaries."""

    # Raw residual is the difference between reconstructed and applied forces, 
    # with shape (timesteps, slices)
    raw_residual = reconstructed_force - applied_longitudinal_force[:, np.newaxis]
    # Normalised residual is relative to applied force, FRE(t) = (F_R(t) - F_A(t)) / F_A(t)
    relative_residual = _compute_relative_force_residual(
        raw_residual,
        applied_longitudinal_force,
    )

    # Weighted RMS across time for each slice, shape (slices,)
    # equation: sqrt(sum(temporal_weights * relative_residual^2))
    weighted_temporal_rms = np.sqrt(
        np.nansum(
            np.asarray(temporal_weights, dtype=np.float64)[:, np.newaxis]
            * relative_residual**2,
            axis=0,
        )
    )

    # Weighted RMS across time and slices, scalar
    # equation: sqrt(sum(temporal_weights * spatial_weights * relative_residual^2))
    weighted_spatiotemporal_rms = float(
        np.sqrt(
            np.nansum(
                np.asarray(temporal_weights, dtype=np.float64)[:, np.newaxis]
                * np.asarray(spatial_weights, dtype=np.float64)[np.newaxis, :]
                * relative_residual**2
            )
        )
    )

    return ForceReconstructionErrorResult(
        metric_result=_build_slice_metric_result(
            raw_residual=raw_residual,
            reconstructed_force=reconstructed_force,
            applied_longitudinal_force=applied_longitudinal_force,
            temporal_weights=temporal_weights,
            spatial_weights=spatial_weights,
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
    """
    Normalise a weight array to sum to 1.0, replacing non-finite or negative weights
    with zero. If the total weight is zero, return a uniform distribution.
    """
    resolved = np.asarray(raw_weights, dtype=np.float64)
    resolved = np.where(np.isfinite(resolved) & (resolved > 0.0), resolved, 0.0)
    total = float(np.sum(resolved))
    if total <= 0.0:
        return np.full(resolved.shape, 1.0 / resolved.size, dtype=np.float64)
    return resolved / total


def _compute_relative_force_residual(
    raw_residual: npt.NDArray[np.float64],
    applied_longitudinal_force: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute relative force-reconstruction error.

    ``FRE = (F_R - F_A) / F_A``.  Zero-force steps carry no useful
    information and are represented as NaN so zero temporal weight can safely
    exclude them from weighted RMS aggregation.
    """

    applied_force = np.asarray(applied_longitudinal_force, dtype=np.float64)
    raw_residual = np.asarray(raw_residual, dtype=np.float64)

    # If applied_force is 1D and raw_residual is 2D (timesteps x slices), 
    # broadcast applied_force to match shape for element-wise division
    if raw_residual.ndim > applied_force.ndim:
        applied_force = applied_force[:, np.newaxis]

    # Compute normalised residual as FRE(t) = (F_R(t) - F_A(t)) / F_A(t),
    # while handling division by zero by setting those entries to NaN
    normalised_residual = np.divide(
        raw_residual,
        applied_force,
        out=np.full_like(raw_residual, np.nan),
        where=np.abs(applied_force) > np.finfo(np.float64).eps,
    )

    return normalised_residual


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


@dataclass(slots=True, frozen=True)
class ForceIntegrationDomainCorrection:
    """Per-slice FRE area correction and its diagnostic provenance."""

    scale_factors: npt.NDArray[np.float64]
    measured_areas: npt.NDArray[np.float64]
    target_areas: npt.NDArray[np.float64]
    target_widths: npt.NDArray[np.float64]
    enabled: bool


def resolve_force_integration_domain_correction(
    *,
    slice_partition: SliceAreaPartition,
    specimen_geometry: SpecimenGeometry,
    valid_global_points: npt.NDArray[np.bool_],
) -> ForceIntegrationDomainCorrection:
    """Resolve optional physical-domain scaling for each FRE slice.

    The measured stress integral is multiplied by ``target area / measured
    area``.  This assumes the measured slice-average stress represents the
    omitted edge strip; it never creates strain or stress samples outside the
    measured ROI.
    """

    target_roi = specimen_geometry.force_reconstruction_region_of_interest
    if target_roi is None:
        # Keep the disabled path both numerically and computationally
        # equivalent to the historical force operator.
        return ForceIntegrationDomainCorrection(
            np.ones(slice_partition.num_slices, dtype=np.float64),
            slice_partition.areas.copy(),
            slice_partition.areas.copy(),
            slice_partition.widths.copy(),
            False,
        )

    measured_areas = np.zeros(slice_partition.num_slices, dtype=np.float64)
    for slice_index, (point_indices, area_integral_weights) in enumerate(
        zip(
            slice_partition.slice_force_point_indices,
            slice_partition.slice_force_point_area_integral_weights,
            strict=True,
        )
    ):
        _, filtered_weights = _filter_operator_points(
            point_indices,
            area_integral_weights,
            valid_global_points,
        )
        measured_areas[slice_index] = (
            float(np.sum(filtered_weights)) * slice_partition.spans[slice_index]
        )

    target_areas = calculate_roi_slice_areas(
        target_roi,
        axis=slice_partition.axis,
        boundaries=slice_partition.boundaries,
    )
    missing_measured_support = (
        (target_areas > np.finfo(np.float64).eps)
        & (measured_areas <= np.finfo(np.float64).eps)
    )
    if np.any(missing_measured_support):
        indices = np.flatnonzero(missing_measured_support).tolist()
        raise ValueError(
            "The physical FRE ROI contains slices with no measured stress "
            f"support: {indices}. The correction cannot extrapolate an empty slice."
        )
    scale_factors = np.divide(
        target_areas,
        measured_areas,
        out=np.ones_like(target_areas),
        where=measured_areas > np.finfo(np.float64).eps,
    )
    if not np.all(np.isfinite(scale_factors)) or np.any(scale_factors <= 0.0):
        raise ValueError("Physical FRE ROI produced non-positive or non-finite area scaling.")
    target_widths = np.divide(
        target_areas,
        slice_partition.spans,
        out=np.zeros_like(target_areas),
        where=slice_partition.spans > 0.0,
    )
    return ForceIntegrationDomainCorrection(
        scale_factors,
        measured_areas,
        target_areas,
        target_widths,
        True,
    )


@dataclass(slots=True, init=False)
class SliceWiseForceReconstructionMetric(IMetric):
    """Area-based slice force reconstruction metric.

    The reconstructed slice force is the discrete analogue of

        F_r = h / L_r * integral_{S_r} sigma dA

    where the area integral is approximated by precomputed overlap areas
    between each slice region and the native DIC support cells.
    """

    support: SupportSlice

    def __init__(
        self,
        support: SupportSlice | None = None,
        slice_partition: SliceAreaPartition | None = None,
        slice_config: SliceConfig | None = None,
    ) -> None:
        if support is None:
            support = SupportSlice(
                slice_partition=slice_partition,
                slice_config=slice_config,
            )
        elif slice_partition is not None or slice_config is not None:
            raise ValueError(
                "Provide either support or slice_partition/slice_config."
            )
        self.support = support

    @property
    def slice_partition(self) -> SliceAreaPartition | None:
        return self.support.slice_partition

    @property
    def slice_config(self) -> SliceConfig | None:
        return self.support.slice_config

    def set_support(
        self,
        support: SupportSlice,
    ) -> None:
        self.support = support

    def initialise_slice_partition(
        self,
        specimen_geometry: SpecimenGeometry,
    ) -> None:
        self.support.prepare_from_specimen_geometry(specimen_geometry)

    def initialise(
        self,
        experiment_data: ExperimentData,
    ) -> None:
        self.initialise_slice_partition(experiment_data.specimen_geometry)

    def evaluate(
        self,
        stress: npt.NDArray[np.float64],
        constitutive_law: IConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
        experiment_data: ExperimentData,
    ) -> MetricResult:
        if self.slice_partition is None:
            raise RuntimeError("Slice partition has not been resolved.")
        return self.evaluate_force_recon_error(stress, experiment_data).metric_result

    def evaluate_local(
        self,
        stress: npt.NDArray[np.float64],
        experiment_data: ExperimentData,
        local_slice_data: LocalSliceData,
    ) -> MetricResult:
        """Return the local slice residual plus weighting metadata."""

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
                * local_slice_data.force_integration_scale
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
        if self.slice_partition is None:
            raise RuntimeError("Slice partition has not been resolved.")
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
        domain_correction = resolve_force_integration_domain_correction(
            slice_partition=self.slice_partition,
            specimen_geometry=experiment_data.specimen_geometry,
            valid_global_points=valid_global_points,
        )

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
                thickness
                * domain_correction.scale_factors[slice_index]
                * np.sum(stress_component[:, point_indices] * point_area_integral_weights[np.newaxis, :], axis=1)
            )

        # Weight temporal FRE contributions by applied-force
        # squared, reducing sensitivity to low-load frames.
        temporal_weights = compute_force_temporal_weights(applied_longitudinal_force)
        spatial_weights = _normalise_weights(domain_correction.target_widths)

        result = build_force_reconstruction_error_result(
            reconstructed_force=reconstructed_force,
            applied_longitudinal_force=applied_longitudinal_force,
            temporal_weights=temporal_weights,
            spatial_weights=spatial_weights,
        )
        assert result.metric_result.additional_fields is not None
        result.metric_result.additional_fields.update({
            "force_integration_domain_correction_enabled": domain_correction.enabled,
            "force_integration_scale_factors": domain_correction.scale_factors,
            "force_integration_measured_areas": domain_correction.measured_areas,
            "force_integration_target_areas": domain_correction.target_areas,
            "force_integration_measured_widths": np.divide(
                domain_correction.measured_areas,
                self.slice_partition.spans,
                out=np.zeros_like(domain_correction.measured_areas),
                where=self.slice_partition.spans > 0.0,
            ),
            "force_integration_target_widths": domain_correction.target_widths,
            "force_integration_represented_fractions": np.divide(
                domain_correction.measured_areas,
                domain_correction.target_areas,
                out=np.ones_like(domain_correction.measured_areas),
                where=domain_correction.target_areas > np.finfo(np.float64).eps,
            ),
        })
        return result

    def normalised_residual_stress_adjoint(
        self,
        cotangent: npt.NDArray[np.float64],
        experiment_data: ExperimentData,
    ) -> npt.NDArray[np.float64]:
        """Back-propagate an FRE cotangent to the stress field."""
        if self.slice_partition is None:
            raise RuntimeError("Slice partition has not been resolved.")
        timesteps, _, rows, columns = experiment_data.strain.shape
        expected_shape = (timesteps, self.slice_partition.num_slices)
        if cotangent.shape != expected_shape:
            raise ValueError(f"Expected cotangent shape {expected_shape}.")

        axis = self.slice_partition.axis
        component = 0 if axis == "x" else 1
        applied_force = _extract_force_component(
            experiment_data.boundary_conditions.force,
            axis,
        )
        force_cotangent = np.divide(
            np.nan_to_num(cotangent, nan=0.0),
            applied_force[:, np.newaxis],
            out=np.zeros_like(cotangent, dtype=np.float64),
            where=np.abs(applied_force[:, np.newaxis]) > np.finfo(np.float64).eps,
        )
        result = np.zeros((timesteps, 3, rows, columns), dtype=np.float64)
        result_flat = result[:, component].reshape(timesteps, -1)
        finite_points = np.all(np.isfinite(experiment_data.strain), axis=(0, 1)).ravel()
        domain_correction = resolve_force_integration_domain_correction(
            slice_partition=self.slice_partition,
            specimen_geometry=experiment_data.specimen_geometry,
            valid_global_points=finite_points,
        )
        thickness = float(experiment_data.specimen_geometry.thickness)
        for slice_index, (point_indices, area_weights) in enumerate(
            zip(
                self.slice_partition.slice_force_point_indices,
                self.slice_partition.slice_force_point_area_integral_weights,
                strict=True,
            )
        ):
            point_indices, area_weights = _filter_operator_points(
                point_indices,
                area_weights,
                finite_points,
            )
            result_flat[:, point_indices] += (
                thickness
                * domain_correction.scale_factors[slice_index]
                * force_cotangent[:, slice_index, np.newaxis]
                * area_weights[np.newaxis]
            )
        return result


def compute_force_temporal_weights(
    applied_longitudinal_force: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return normalised applied-force-squared temporal weights.
    
    F^2 weighting reduces sensitivity to low-load frames. 
    """
    force = np.asarray(applied_longitudinal_force, dtype=np.float64)
    weights = force**2
    if not np.any(weights > 0.0):
        raise ValueError(
            "Force reconstruction error requires at least one non-zero "
            "applied-force timestep."
        )
    return _normalise_weights(weights)
