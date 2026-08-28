"""Combined scalar force-reconstruction error and equilibrium-gap objective.

The formulation is based on the refined identification objective in Hamill
et al. (2026).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
import time

import numpy as np
import numpy.typing as npt

from pyvale.vfm.equilibriumgapaggregation import (
    extract_equilibrium_gap_temporal_rms,
)
from pyvale.vfm.metric import MetricResult
from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.metricequilibriumgap import EquilibriumGapMetric
from pyvale.vfm.metricsliceforce import SliceWiseForceReconstructionMetric
from pyvale.vfm.objectivefunc import IScalarObjectiveFunction
from pyvale.vfm.objectivefuncfreandegi import (
    calculate_force_reconstruction_spatiotemporal_rms,
)
from pyvale.vfm.spatialweighting import (
    SensitivitySpatialWeightingConfig,
    SensitivitySpatialWeights,
    calculate_sensitivity_spatial_weights,
)


class CombinedObjectiveBaselineMode(StrEnum):
    """Source of the FRE and EGI normalisation values."""

    UNIT = "unit"
    MANUAL = "manual"
    PRIOR_PHASE = "prior_phase"


@dataclass(frozen=True, slots=True)
class CombinedObjectiveBaseline:
    """How a combined-objective baseline is selected.

    ``UNIT`` leaves metric values unnormalised. ``MANUAL`` uses caller-supplied
    EGI and FRE values. ``PRIOR_PHASE`` calculates both from the final stress
    of an earlier identification phase.
    """

    mode: CombinedObjectiveBaselineMode = CombinedObjectiveBaselineMode.UNIT
    egi_values: tuple[float, ...] | None = None
    force_value: float | None = None
    phase_index: int | None = None

    @classmethod
    def unit(cls) -> "CombinedObjectiveBaseline":
        return cls()

    @classmethod
    def manual(
        cls,
        egi_values: Sequence[float],
        force_value: float,
    ) -> "CombinedObjectiveBaseline":
        return cls(
            mode=CombinedObjectiveBaselineMode.MANUAL,
            egi_values=tuple(float(value) for value in egi_values),
            force_value=float(force_value),
        )

    @classmethod
    def prior_phase(cls, phase_index: int) -> "CombinedObjectiveBaseline":
        return cls(
            mode=CombinedObjectiveBaselineMode.PRIOR_PHASE,
            phase_index=phase_index,
        )


@dataclass(slots=True, frozen=True)
class CombinedForceAndEquilibriumGapObjectiveResult:
    """Decomposed diagnostics for one objective evaluation."""

    total_cost: float
    equilibrium_gap_cost: float
    force_cost: float
    egi_scalars: npt.NDArray[np.float64]
    egi_baselines: npt.NDArray[np.float64]
    egi_window_weights: npt.NDArray[np.float64]
    force_scalar: float
    force_baseline: float
    force_weight: float


@dataclass(slots=True, frozen=True)
class CombinedObjectiveResidualCotangents:
    """Derivatives of the scalar objective with respect to signed residuals."""

    equilibrium_gap: tuple[npt.NDArray[np.float64], ...]
    force: npt.NDArray[np.float64]


class CombinedForceAndEquilibriumGapObjective(IScalarObjectiveFunction):
    """Scalar objective combining normalised EGI and FRE contributions.

    ``phi = (1-lambda) sum(gamma_k * EGI_k / a_k)
    + lambda * FRE / b``.

    The EGI scalars are evaluated independently.  A combined EGI map remains
    useful for seeding basis centres, but is deliberately not used here: the
    RMS of a combined map is not equivalent to this scalar aggregation. By
    default ``a_k`` and ``b`` are one; callers may instead provide values or
    resolve them from a completed earlier identification phase.
    """

    def __init__(
        self,
        *,
        force_weight: float = 0.1,
        egi_window_weights: Sequence[float] | npt.NDArray[np.float64],
        baseline: CombinedObjectiveBaseline | None = None,
        egi_baseline_values: Sequence[float] | npt.NDArray[np.float64] | None = None,
        force_baseline_value: float | None = None,
        spatial_weighting: SensitivitySpatialWeightingConfig | None = None,
    ) -> None:
        if not 0.0 <= force_weight <= 1.0:
            raise ValueError("force_weight must lie in [0, 1].")
        if spatial_weighting is not None and not isinstance(
            spatial_weighting,
            SensitivitySpatialWeightingConfig,
        ):
            raise TypeError(
                "spatial_weighting must be a "
                "SensitivitySpatialWeightingConfig or None."
            )
        if baseline is not None and (
            egi_baseline_values is not None or force_baseline_value is not None
        ):
            raise ValueError(
                "Provide either baseline or egi_baseline_values and "
                "force_baseline_value, not both."
            )
        if baseline is None and (
            egi_baseline_values is not None or force_baseline_value is not None
        ):
            if egi_baseline_values is None or force_baseline_value is None:
                raise ValueError(
                    "egi_baseline_values and force_baseline_value must be "
                    "provided together."
                )
            baseline = CombinedObjectiveBaseline.manual(
                egi_baseline_values,
                force_baseline_value,
            )
        self.force_weight = float(force_weight)
        self.spatial_weighting = spatial_weighting
        self.baseline = baseline or CombinedObjectiveBaseline.unit()
        self.egi_window_weights = _normalised_weights(egi_window_weights)
        self.resolved_egi_baseline_values: npt.NDArray[np.float64] | None = None
        self.resolved_force_baseline_value: float | None = None
        self.resolved_spatial_weights: SensitivitySpatialWeights | None = None
        self.spatial_weighting_runtime_seconds: float | None = None
        self._validate_baseline_configuration()
        self.last_result: CombinedForceAndEquilibriumGapObjectiveResult | None = None

    def evaluate(self, metric_results: list[MetricResult]) -> float:
        force_results, egi_results = _split_metric_results(metric_results)
        if len(force_results) != 1:
            raise ValueError("Objective requires exactly one FRE metric result.")
        egi_spatial_weights = self._egi_spatial_weights_for_results(egi_results)
        egi_scalars = np.asarray(
            [
                _extract_egi_scalar(result, spatial_weights=spatial_weights)
                for result, spatial_weights in zip(
                    egi_results,
                    egi_spatial_weights,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
        force_scalar = calculate_force_reconstruction_spatiotemporal_rms(
            force_results[0],
            spatial_weights=self._force_spatial_weights(),
        )
        egi_baselines, force_baseline = self._resolve_baselines(
            egi_scalars,
            force_scalar,
        )
        equilibrium_gap_cost = float(
            np.sum(
                self.egi_window_weights
                * egi_scalars
                / egi_baselines
            )
        )
        force_cost = float(force_scalar / force_baseline)
        total_cost = (
            (1.0 - self.force_weight) * equilibrium_gap_cost
            + self.force_weight * force_cost
        )
        self.last_result = CombinedForceAndEquilibriumGapObjectiveResult(
            total_cost=float(total_cost),
            equilibrium_gap_cost=equilibrium_gap_cost,
            force_cost=force_cost,
            egi_scalars=egi_scalars,
            egi_baselines=egi_baselines.copy(),
            egi_window_weights=self.egi_window_weights.copy(),
            force_scalar=float(force_scalar),
            force_baseline=force_baseline,
            force_weight=self.force_weight,
        )
        return float(total_cost)

    def residual_cotangents(
        self,
        metric_results: list[MetricResult],
    ) -> CombinedObjectiveResidualCotangents:
        """Differentiate the exact scalar aggregation wrt EGI and FRE fields."""
        force_results, egi_results = _split_metric_results(metric_results)
        if len(force_results) != 1:
            raise ValueError("Objective requires exactly one FRE metric result.")
        self.evaluate(metric_results)
        assert self.last_result is not None

        egi_spatial_weights = self._egi_spatial_weights_for_results(egi_results)
        egi_cotangents = tuple(
            _egi_scalar_cotangent(result, spatial_weights=spatial_weights)
            * (
                (1.0 - self.force_weight)
                * self.egi_window_weights[index]
                / self.last_result.egi_baselines[index]
            )
            for index, (result, spatial_weights) in enumerate(
                zip(egi_results, egi_spatial_weights, strict=True)
            )
        )
        force_cotangent = _force_scalar_cotangent(
            force_results[0],
            spatial_weights=self._force_spatial_weights(),
        ) * (self.force_weight / self.last_result.force_baseline)
        return CombinedObjectiveResidualCotangents(
            equilibrium_gap=egi_cotangents,
            force=force_cotangent,
        )

    def resolve_from_prior_phase(self, metric_results: list[MetricResult]) -> None:
        """Set baseline values from metrics evaluated on a referenced phase."""

        if self.baseline.mode is not CombinedObjectiveBaselineMode.PRIOR_PHASE:
            return
        force_results, egi_results = _split_metric_results(metric_results)
        if len(force_results) != 1:
            raise ValueError("Objective requires exactly one FRE metric result.")
        egi_spatial_weights = self._egi_spatial_weights_for_results(egi_results)
        egi_values = np.asarray(
            [
                _extract_egi_scalar(result, spatial_weights=spatial_weights)
                for result, spatial_weights in zip(
                    egi_results,
                    egi_spatial_weights,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
        force_value = calculate_force_reconstruction_spatiotemporal_rms(
            force_results[0],
            spatial_weights=self._force_spatial_weights(),
        )
        self._set_resolved_baselines(egi_values, force_value)

    def resolve_spatial_weights(
        self,
        *,
        constitutive_law: IConstitutiveLaw,
        parameter_maps: dict[str, npt.NDArray[np.float64]],
        active_parameter_names: Sequence[str],
        metrics: Sequence[object],
        experiment_data: ExperimentData,
    ) -> None:
        """Resolve and freeze sensitivity-derived weights for this phase."""

        if self.spatial_weighting is None:
            return
        force_metrics = [
            metric
            for metric in metrics
            if isinstance(metric, SliceWiseForceReconstructionMetric)
        ]
        egi_metrics = [
            metric
            for metric in metrics
            if isinstance(metric, EquilibriumGapMetric)
        ]
        if len(force_metrics) != 1:
            raise ValueError(
                "Sensitivity spatial weighting requires exactly one "
                "SliceWiseForceReconstructionMetric."
            )
        if not egi_metrics:
            raise ValueError(
                "Sensitivity spatial weighting requires at least one "
                "EquilibriumGapMetric."
            )

        started_at = time.perf_counter()
        stress_reference = constitutive_law.calculate_stress(
            experiment_data.strain,
            parameter_maps,
        )
        self.resolved_spatial_weights = calculate_sensitivity_spatial_weights(
            experiment_data.strain,
            stress_reference,
            constitutive_law,
            parameter_maps,
            list(active_parameter_names),
            egi_metrics,
            force_metrics[0],
            experiment_data,
            self.spatial_weighting,
        )
        self.spatial_weighting_runtime_seconds = time.perf_counter() - started_at

    def egi_spatial_weights_for(
        self,
        number_of_windows: int,
    ) -> tuple[npt.NDArray[np.float64], ...] | None:
        """Return frozen EGI centre weights, or ``None`` when disabled."""

        if self.spatial_weighting is None:
            return None
        if self.resolved_spatial_weights is None:
            raise ValueError("Sensitivity spatial weights have not been resolved.")
        weights = self.resolved_spatial_weights.equilibrium_gap_weights
        if len(weights) != number_of_windows:
            raise ValueError(
                "Resolved EGI spatial weights do not match the EGI results: "
                f"{len(weights)} vs {number_of_windows}."
            )
        return tuple(weight.copy() for weight in weights)

    def spatial_weighting_diagnostics(self) -> dict[str, object]:
        """Return compact diagnostics without serialising full weight maps."""

        if self.spatial_weighting is None:
            return {"enabled": False}
        if self.resolved_spatial_weights is None:
            return {"enabled": True, "resolved": False}
        return {
            "enabled": True,
            "resolved": True,
            "perturbation_factor": self.spatial_weighting.perturbation_factor,
            "weight_floor": self.spatial_weighting.weight_floor,
            "scaling_percentile": self.spatial_weighting.scaling_percentile,
            "parameter_names": list(self.resolved_spatial_weights.parameter_names),
            "runtime_seconds": self.spatial_weighting_runtime_seconds,
            "force_weights": _array_summary(self.resolved_spatial_weights.force_weights),
            "egi_weights": [
                _array_summary(weights)
                for weights in self.resolved_spatial_weights.equilibrium_gap_weights
            ],
        }

    def baseline_diagnostics(self) -> dict[str, object]:
        """Return the configured source and resolved values for result metadata."""

        return {
            "mode": self.baseline.mode.value,
            "phase_index": self.baseline.phase_index,
            "egi_values": (
                None
                if self.resolved_egi_baseline_values is None
                else self.resolved_egi_baseline_values.tolist()
            ),
            "force_value": self.resolved_force_baseline_value,
        }

    def egi_baselines_for(
        self,
        number_of_windows: int,
    ) -> npt.NDArray[np.float64]:
        """Return the configured EGI baselines for map aggregation."""

        if number_of_windows < 1:
            raise ValueError("number_of_windows must be at least one.")
        if self.baseline.mode is CombinedObjectiveBaselineMode.UNIT:
            return np.ones(number_of_windows, dtype=np.float64)
        if self.resolved_egi_baseline_values is None:
            raise ValueError("Combined-objective EGI baselines must be resolved first.")
        if self.resolved_egi_baseline_values.shape != (number_of_windows,):
            raise ValueError(
                "Objective requires one EGI baseline per EGI result: "
                f"{self.resolved_egi_baseline_values.shape} vs "
                f"({number_of_windows},)."
            )
        return self.resolved_egi_baseline_values.copy()

    def _egi_spatial_weights_for_results(
        self,
        egi_results: Sequence[MetricResult],
    ) -> tuple[npt.NDArray[np.float64] | None, ...]:
        weights = self.egi_spatial_weights_for(len(egi_results))
        if weights is None:
            return (None,) * len(egi_results)
        return weights

    def _force_spatial_weights(self) -> npt.NDArray[np.float64] | None:
        if self.spatial_weighting is None:
            return None
        if self.resolved_spatial_weights is None:
            raise ValueError("Sensitivity spatial weights have not been resolved.")
        return self.resolved_spatial_weights.force_weights

    def _validate_baseline_configuration(self) -> None:
        if not isinstance(self.baseline.mode, CombinedObjectiveBaselineMode):
            raise ValueError("baseline.mode must be a CombinedObjectiveBaselineMode.")
        match self.baseline.mode:
            case CombinedObjectiveBaselineMode.UNIT:
                if any(
                    value is not None
                    for value in (
                        self.baseline.egi_values,
                        self.baseline.force_value,
                        self.baseline.phase_index,
                    )
                ):
                    raise ValueError("A unit baseline cannot specify values or a phase.")
            case CombinedObjectiveBaselineMode.MANUAL:
                if self.baseline.phase_index is not None:
                    raise ValueError("A manual baseline cannot specify a phase.")
                if self.baseline.egi_values is None or self.baseline.force_value is None:
                    raise ValueError("A manual baseline requires EGI and FRE values.")
                self._set_resolved_baselines(
                    np.asarray(self.baseline.egi_values, dtype=np.float64),
                    self.baseline.force_value,
                )
            case CombinedObjectiveBaselineMode.PRIOR_PHASE:
                if self.baseline.phase_index is None:
                    raise ValueError("A prior-phase baseline requires phase_index.")
                if self.baseline.egi_values is not None or self.baseline.force_value is not None:
                    raise ValueError("A prior-phase baseline cannot specify manual values.")

    def _resolve_baselines(
        self,
        egi_values: npt.NDArray[np.float64],
        force_value: float,
    ) -> tuple[npt.NDArray[np.float64], float]:
        if self.baseline.mode is CombinedObjectiveBaselineMode.UNIT:
            self._set_resolved_baselines(np.ones_like(egi_values), 1.0)
        if self.resolved_egi_baseline_values is None or self.resolved_force_baseline_value is None:
            raise ValueError(
                "Prior-phase baselines have not been resolved. "
                "Run this objective through identification first."
            )
        if self.egi_window_weights.shape != egi_values.shape:
            raise ValueError(
                "Objective requires one EGI window weight per EGI result: "
                f"{self.egi_window_weights.shape} vs {egi_values.shape}."
            )
        if self.resolved_egi_baseline_values.shape != egi_values.shape:
            raise ValueError(
                "Objective requires one EGI baseline per EGI result: "
                f"{self.resolved_egi_baseline_values.shape} vs {egi_values.shape}."
            )
        return self.resolved_egi_baseline_values, self.resolved_force_baseline_value

    def _set_resolved_baselines(
        self,
        egi_values: npt.NDArray[np.float64],
        force_value: float,
    ) -> None:
        self.resolved_egi_baseline_values = _positive_vector(
            egi_values,
            "egi_baseline_values",
        )
        self.resolved_force_baseline_value = _positive_scalar(
            force_value,
            "force_baseline_value",
        )


def infer_egi_window_length_weights(
    metric_results: Sequence[MetricResult],
) -> npt.NDArray[np.float64]:
    """Return EGI window-length weights from EGI metadata.

    The paper uses mean window side length, rather than window area.  For a
    square 29/57 pair this gives weights 29/(29+57), 57/(29+57).
    """

    lengths = []
    for metric_result in metric_results:
        window_size = (metric_result.additional_fields or {}).get("window_size")
        if window_size is None:
            raise ValueError("EGI metric result does not contain window_size metadata.")
        dimensions = np.asarray(window_size, dtype=np.float64)
        if dimensions.shape != (2,) or np.any(dimensions <= 0.0):
            raise ValueError(f"Invalid EGI window_size metadata {dimensions}.")
        lengths.append(float(np.mean(dimensions)))
    return _normalised_weights(lengths)


def _split_metric_results(
    metric_results: Sequence[MetricResult],
) -> tuple[list[MetricResult], list[MetricResult]]:
    force_results: list[MetricResult] = []
    egi_results: list[MetricResult] = []
    for metric_result in metric_results:
        metadata = metric_result.additional_fields or {}
        if "weighted_spatiotemporal_rms" in metadata and "window_size" in metadata:
            egi_results.append(metric_result)
        elif "reconstructed_force" in metadata and "normalised_residual" in metadata:
            force_results.append(metric_result)
    return force_results, egi_results


def _extract_egi_scalar(
    metric_result: MetricResult,
    *,
    spatial_weights: npt.NDArray[np.float64] | None = None,
) -> float:
    if spatial_weights is not None:
        temporal_rms = extract_equilibrium_gap_temporal_rms(metric_result)
        spatial = np.asarray(spatial_weights, dtype=np.float64)
        if temporal_rms.shape != spatial.shape:
            raise ValueError(
                "EGI spatial weights do not match the residual map: "
                f"{spatial.shape} vs {temporal_rms.shape}."
            )
        valid_centres = np.isfinite(temporal_rms)
        if (
            np.any(~np.isfinite(spatial[valid_centres]))
            or np.any(spatial[valid_centres] < 0.0)
        ):
            raise ValueError(
                "EGI spatial weights must be finite and non-negative on "
                "valid EGI centres."
            )
        weighted = temporal_rms * np.sqrt(spatial)
        finite = weighted[np.isfinite(weighted)]
        if finite.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean(finite**2)))

    value = (metric_result.additional_fields or {}).get(
        "weighted_spatiotemporal_rms"
    )
    return _non_negative_scalar(value, "EGI weighted_spatiotemporal_rms")


def _egi_scalar_cotangent(
    metric_result: MetricResult,
    *,
    spatial_weights: npt.NDArray[np.float64] | None,
) -> npt.NDArray[np.float64]:
    metadata = metric_result.additional_fields or {}
    gap = np.asarray(metadata.get("normalised_gap", metric_result.residual), dtype=np.float64)
    temporal_weights = np.asarray(metadata["temporal_weights"], dtype=np.float64)
    scalar = _extract_egi_scalar(metric_result, spatial_weights=spatial_weights)
    cotangent = np.zeros_like(gap)
    if scalar <= 0.0:
        return cotangent
    valid = np.isfinite(gap)
    if spatial_weights is None:
        valid_count = int(np.sum(valid))
        cotangent[valid] = (
            gap[valid]
            * np.broadcast_to(temporal_weights[:, None, None], gap.shape)[valid]
            / (scalar * valid_count)
        )
        return cotangent

    spatial = np.asarray(spatial_weights, dtype=np.float64)
    valid_per_centre = np.sum(valid, axis=0)
    valid_centres = valid_per_centre > 0
    centre_count = int(np.sum(valid_centres))
    denominator = scalar * centre_count * valid_per_centre
    scale = np.divide(
        spatial,
        denominator,
        out=np.zeros_like(spatial),
        where=denominator > 0.0,
    )
    cotangent = (
        np.nan_to_num(gap, nan=0.0)
        * temporal_weights[:, None, None]
        * scale[None]
    )
    cotangent[~valid] = 0.0
    return cotangent


def _force_scalar_cotangent(
    metric_result: MetricResult,
    *,
    spatial_weights: npt.NDArray[np.float64] | None,
) -> npt.NDArray[np.float64]:
    metadata = metric_result.additional_fields or {}
    residual = np.asarray(
        metadata.get("normalised_residual", metric_result.residual),
        dtype=np.float64,
    )
    temporal = np.asarray(metadata.get("temporal_weights", 1.0), dtype=np.float64)
    resolved_spatial = (
        metadata.get("spatial_weights")
        if spatial_weights is None
        else spatial_weights
    )
    spatial = np.asarray(
        1.0 if resolved_spatial is None else resolved_spatial,
        dtype=np.float64,
    )
    scalar = calculate_force_reconstruction_spatiotemporal_rms(
        metric_result,
        spatial_weights=spatial_weights,
    )
    if scalar <= 0.0:
        return np.zeros_like(residual)
    weights = temporal[:, None] * spatial
    result = np.nan_to_num(residual, nan=0.0) * weights / scalar
    result[~np.isfinite(residual)] = 0.0
    return result


def _array_summary(values: npt.NDArray[np.float64]) -> dict[str, object]:
    resolved = np.asarray(values, dtype=np.float64)
    finite = resolved[np.isfinite(resolved)]
    return {
        "shape": list(resolved.shape),
        "finite_count": int(finite.size),
        "min": None if finite.size == 0 else float(np.min(finite)),
        "mean": None if finite.size == 0 else float(np.mean(finite)),
        "max": None if finite.size == 0 else float(np.max(finite)),
    }


def _positive_scalar(value: object, name: str) -> float:
    scalar = float(value)  # type: ignore[arg-type]
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return scalar


def _non_negative_scalar(value: object, name: str) -> float:
    scalar = float(value)  # type: ignore[arg-type]
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return scalar


def _positive_vector(value: Sequence[float] | npt.NDArray[np.float64], name: str) -> npt.NDArray[np.float64]:
    resolved = np.asarray(value, dtype=np.float64)
    if resolved.ndim != 1 or resolved.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence.")
    if np.any(~np.isfinite(resolved)) or np.any(resolved <= 0.0):
        raise ValueError(f"{name} must contain finite values greater than zero.")
    return resolved


def _normalised_weights(value: Sequence[float] | npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    resolved = _positive_vector(value, "egi_window_weights")
    return resolved / float(np.sum(resolved))
