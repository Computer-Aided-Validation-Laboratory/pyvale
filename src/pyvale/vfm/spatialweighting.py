"""Sensitivity-derived spatial weights for VFM residual metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.metricsbvf import (
    StressSensitivity,
    calculate_parameter_stress_sensitivities,
)
from pyvale.vfm.metricequilibriumgap import EquilibriumGapMetric
from pyvale.vfm.metricsliceforce import SliceWiseForceReconstructionMetric


@dataclass(slots=True, frozen=True)
class SensitivitySpatialWeightingConfig:
    """Controls for robustly converting metric sensitivity into weights."""

    perturbation_factor: float = 0.15
    weight_floor: float = 0.1
    scaling_percentile: float = 95.0

    def __post_init__(self) -> None:
        if not 0.0 < self.perturbation_factor < 1.0:
            raise ValueError("perturbation_factor must lie in (0, 1).")
        if not 0.0 < self.weight_floor <= 1.0:
            raise ValueError("weight_floor must lie in (0, 1].")
        if not 0.0 < self.scaling_percentile <= 100.0:
            raise ValueError("scaling_percentile must lie in (0, 100].")


@dataclass(slots=True, frozen=True)
class SensitivitySpatialWeights:
    """Resolved weights and parameter-specific activity diagnostics."""

    parameter_names: tuple[str, ...]
    equilibrium_gap_weights: tuple[npt.NDArray[np.float64], ...]
    force_weights: npt.NDArray[np.float64]
    equilibrium_gap_parameter_activity: tuple[
        dict[str, npt.NDArray[np.float64]], ...
    ]
    force_parameter_activity: dict[str, npt.NDArray[np.float64]]


def calculate_sensitivity_spatial_weights(
    strain: npt.NDArray[np.float64],
    stress_reference: npt.NDArray[np.float64],
    constitutive_law: IConstitutiveLaw,
    parameter_maps: dict[str, npt.NDArray[np.float64]],
    active_parameter_names: tuple[str, ...] | list[str],
    equilibrium_gap_metrics: Sequence[EquilibriumGapMetric],
    force_metric: SliceWiseForceReconstructionMetric,
    experiment_data: ExperimentData,
    config: SensitivitySpatialWeightingConfig | None = None,
) -> SensitivitySpatialWeights:
    """Calculate parameter sensitivities and project them into metric weights."""

    resolved_config = config or SensitivitySpatialWeightingConfig()
    sensitivities = calculate_parameter_stress_sensitivities(
        strain,
        stress_reference,
        constitutive_law,
        parameter_maps,
        active_parameter_names,
        perturbation_factor=resolved_config.perturbation_factor,
    )
    return resolve_sensitivity_spatial_weights(
        sensitivities,
        equilibrium_gap_metrics,
        force_metric,
        experiment_data,
        resolved_config,
    )


def resolve_sensitivity_spatial_weights(
    stress_sensitivities: Mapping[str, StressSensitivity],
    equilibrium_gap_metrics: Sequence[EquilibriumGapMetric],
    force_metric: SliceWiseForceReconstructionMetric,
    experiment_data: ExperimentData,
    config: SensitivitySpatialWeightingConfig | None = None,
) -> SensitivitySpatialWeights:
    """Project stress sensitivities through EGI and FRE metric operators."""

    resolved_config = config or SensitivitySpatialWeightingConfig()
    if not stress_sensitivities:
        raise ValueError("At least one stress sensitivity is required.")
    if not equilibrium_gap_metrics:
        raise ValueError("At least one equilibrium-gap metric is required.")
    if force_metric.slice_partition is None:
        raise RuntimeError("Force metric must be initialised before resolving weights.")

    parameter_names = tuple(stress_sensitivities)
    egi_parameter_activity: list[dict[str, npt.NDArray[np.float64]]] = []
    egi_weights: list[npt.NDArray[np.float64]] = []
    for metric in equilibrium_gap_metrics:
        activities = {
            name: _equilibrium_gap_activity(metric, sensitivity)
            for name, sensitivity in stress_sensitivities.items()
        }
        egi_parameter_activity.append(activities)
        egi_weights.append(
            _build_weights(
                activities,
                floor=resolved_config.weight_floor,
                percentile=resolved_config.scaling_percentile,
                geometry_weights=None,
                normalisation="mean",
            )
        )

    force_parameter_activity = {
        name: _force_activity(force_metric, sensitivity, experiment_data)
        for name, sensitivity in stress_sensitivities.items()
    }
    force_weights = _build_weights(
        force_parameter_activity,
        floor=resolved_config.weight_floor,
        percentile=resolved_config.scaling_percentile,
        geometry_weights=force_metric.slice_partition.widths,
        normalisation="sum",
    )
    return SensitivitySpatialWeights(
        parameter_names=parameter_names,
        equilibrium_gap_weights=tuple(egi_weights),
        force_weights=force_weights,
        equilibrium_gap_parameter_activity=tuple(egi_parameter_activity),
        force_parameter_activity=force_parameter_activity,
    )


def _equilibrium_gap_activity(
    metric: EquilibriumGapMetric,
    sensitivity: StressSensitivity,
) -> npt.NDArray[np.float64]:
    result = metric.evaluate_equilibrium_gap(sensitivity.total)
    if result.weighted_temporal_rms is None:
        raise ValueError("Equilibrium-gap metric did not produce a temporal RMS map.")
    return np.asarray(result.weighted_temporal_rms, dtype=np.float64) ** 2


def _force_activity(
    metric: SliceWiseForceReconstructionMetric,
    sensitivity: StressSensitivity,
    experiment_data: ExperimentData,
) -> npt.NDArray[np.float64]:
    metric_result = metric.evaluate_force_recon_error(
        sensitivity.total,
        experiment_data,
    ).metric_result
    metadata = metric_result.additional_fields or {}
    reconstructed_force = np.asarray(metadata["reconstructed_force"], dtype=np.float64)
    applied_force = np.asarray(metadata["applied_longitudinal_force"], dtype=np.float64)
    temporal_weights = np.asarray(metadata["temporal_weights"], dtype=np.float64)
    relative_response = np.divide(
        reconstructed_force,
        applied_force[:, np.newaxis],
        out=np.full_like(reconstructed_force, np.nan),
        where=np.abs(applied_force[:, np.newaxis]) > np.finfo(np.float64).eps,
    )
    return np.nansum(
        temporal_weights[:, np.newaxis] * relative_response**2,
        axis=0,
    )


def _build_weights(
    parameter_activity: Mapping[str, npt.NDArray[np.float64]],
    *,
    floor: float,
    percentile: float,
    geometry_weights: npt.NDArray[np.float64] | None,
    normalisation: str,
) -> npt.NDArray[np.float64]:
    scaled = np.stack(
        [
            _scale_activity(activity, percentile)
            for activity in parameter_activity.values()
        ],
        axis=0,
    )
    valid = np.any(np.isfinite(scaled), axis=0)
    combined = np.max(np.where(np.isfinite(scaled), scaled, 0.0), axis=0)
    raw_weights = floor + (1.0 - floor) * combined
    if geometry_weights is not None:
        geometry = np.asarray(geometry_weights, dtype=np.float64)
        if geometry.shape != raw_weights.shape:
            raise ValueError(
                "Geometry and activity weight shapes differ: "
                f"{geometry.shape} vs {raw_weights.shape}."
            )
        if np.any(~np.isfinite(geometry)) or np.any(geometry < 0.0):
            raise ValueError("Geometry weights must be finite and non-negative.")
        raw_weights *= geometry

    weights = np.full(raw_weights.shape, np.nan, dtype=np.float64)
    weights[valid] = raw_weights[valid]
    if normalisation == "mean":
        scale = float(np.mean(weights[valid])) if np.any(valid) else 0.0
    elif normalisation == "sum":
        scale = float(np.sum(weights[valid])) if np.any(valid) else 0.0
    else:
        raise ValueError(f"Unsupported weight normalisation '{normalisation}'.")
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("Spatial weights do not contain positive finite support.")
    weights[valid] /= scale
    return weights


def _scale_activity(
    activity: npt.NDArray[np.float64],
    percentile: float,
) -> npt.NDArray[np.float64]:
    resolved = np.asarray(activity, dtype=np.float64)
    valid = np.isfinite(resolved)
    if np.any(resolved[valid] < 0.0):
        raise ValueError("Sensitivity activity must be non-negative.")
    scaled = np.full(resolved.shape, np.nan, dtype=np.float64)
    positive = resolved[valid & (resolved > 0.0)]
    if positive.size == 0:
        scaled[valid] = 0.0
        return scaled
    scale = float(np.percentile(positive, percentile))
    if not np.isfinite(scale) or scale <= 0.0:
        scaled[valid] = 0.0
        return scaled
    scaled[valid] = np.clip(resolved[valid] / scale, 0.0, 1.0)
    return scaled
