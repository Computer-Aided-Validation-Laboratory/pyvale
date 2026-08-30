"""Hybrid global-closure and material-information scalar objective."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import enum
from typing import Sequence

import numpy as np
import numpy.typing as npt

from pyvale.vfm.materialprojection import (
    NativeDofSensitivityAudit,
    NativeDofSensitivityAuditConfig,
)
from pyvale.vfm.metric import MetricResult
from pyvale.vfm.objectivefunc import IScalarObjectiveFunction
from pyvale.vfm.objectivefunccombinedfreegi import (
    CombinedForceAndEquilibriumGapObjective,
)
from pyvale.vfm.residualfeatures import (
    ResidualFeatureResult,
    coherent_rms,
    projected_rms,
    smooth_positive_part,
    weighted_cvar_abs,
    weighted_rms,
)
from pyvale.vfm.solvepreparation import SolvePreparationContext


class MaterialFeatureReduction(enum.StrEnum):
    RMS = "rms"
    CVAR_ABS = "cvar_abs"
    COHERENT_RMS = "coherent_rms"
    PROJECTED_RMS = "projected_rms"


@dataclass(slots=True, frozen=True)
class MaterialFeatureTerm:
    """One semantic residual feature in the material-information block."""

    name: str
    metric_result_index: int
    reduction: MaterialFeatureReduction
    frame_indices: tuple[int, ...] | None = None
    weight: float = 1.0
    quantile: float = 0.90
    sigma_pixels: float | tuple[float, ...] = 2.0
    spatial_axes: tuple[int, ...] = (-2, -1)
    projection_basis: npt.NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Material feature name cannot be empty.")
        if self.metric_result_index < 0:
            raise ValueError("metric_result_index must be non-negative.")
        if not np.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("Material feature weight must be finite and positive.")


@dataclass(slots=True, frozen=True)
class MaterialFeatureReference:
    noise_floor: float
    stage_reference: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.noise_floor) or self.noise_floor < 0.0:
            raise ValueError("noise_floor must be finite and non-negative.")
        if not np.isfinite(self.stage_reference) or self.stage_reference <= self.noise_floor:
            raise ValueError("stage_reference must be finite and exceed noise_floor.")


@dataclass(slots=True, frozen=True)
class MaterialInformationFeatureDiagnostic:
    name: str
    raw_value: float
    noise_floor: float
    stage_reference: float
    normalised_value: float
    weighted_value: float
    valid_count: int
    effective_count: float


@dataclass(slots=True, frozen=True)
class MaterialInformationObjectiveResult:
    total_cost: float
    global_cost: float
    material_cost: float
    material_smooth_max: float
    material_mean: float
    features: tuple[MaterialInformationFeatureDiagnostic, ...]


class MaterialInformationObjective(IScalarObjectiveFunction):
    """Combine mechanical closure with sparse, noise-aware residual features.

    References are deliberately explicit and frozen.  Offline tools can set
    them directly; the identification lifecycle can refresh stage references
    once per fixed-basis solve without changing the optimiser-facing API.
    """

    def __init__(
        self,
        *,
        global_objective: IScalarObjectiveFunction | None,
        feature_terms: Sequence[MaterialFeatureTerm],
        alpha: float = 0.5,
        smooth_max_temperature: float = 0.10,
        mean_fraction: float = 0.10,
        positive_part_temperature: float = 1.0e-3,
        references: dict[str, MaterialFeatureReference] | None = None,
        sensitivity_audit: NativeDofSensitivityAuditConfig | None = None,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be within [0, 1].")
        if global_objective is None and alpha < 1.0:
            raise ValueError("A global objective is required when alpha < 1.")
        if not feature_terms and alpha > 0.0:
            raise ValueError("At least one feature term is required when alpha > 0.")
        names = [term.name for term in feature_terms]
        if len(names) != len(set(names)):
            raise ValueError("Material feature names must be unique.")
        if smooth_max_temperature <= 0.0 or positive_part_temperature <= 0.0:
            raise ValueError("Smooth temperatures must be positive.")
        if not 0.0 <= mean_fraction <= 1.0:
            raise ValueError("mean_fraction must be within [0, 1].")
        self.global_objective = global_objective
        self.feature_terms = tuple(feature_terms)
        self.alpha = float(alpha)
        self.smooth_max_temperature = float(smooth_max_temperature)
        self.mean_fraction = float(mean_fraction)
        self.positive_part_temperature = float(positive_part_temperature)
        self.references = dict(references or {})
        self.sensitivity_audit = sensitivity_audit
        self.last_sensitivity_audit: NativeDofSensitivityAudit | None = None
        self.last_result: MaterialInformationObjectiveResult | None = None
        self._validate_references(require_all=False)

    @property
    def baseline(self):
        """Expose a wrapped combined-objective baseline to the phase runtime."""

        return getattr(self.global_objective, "baseline", None)

    @property
    def spatial_weighting(self):
        return getattr(self.global_objective, "spatial_weighting", None)

    def resolve_from_prior_phase(self, metric_results: list[MetricResult]) -> None:
        resolver = getattr(self.global_objective, "resolve_from_prior_phase", None)
        if resolver is not None:
            resolver(metric_results)

    def resolve_spatial_weights(self, **kwargs) -> None:
        resolver = getattr(self.global_objective, "resolve_spatial_weights", None)
        if resolver is not None:
            resolver(**kwargs)

    def baseline_diagnostics(self) -> dict[str, object]:
        diagnostics = getattr(self.global_objective, "baseline_diagnostics", None)
        return {} if diagnostics is None else diagnostics()

    def spatial_weighting_diagnostics(self) -> dict[str, object]:
        diagnostics = getattr(
            self.global_objective, "spatial_weighting_diagnostics", None
        )
        return {} if diagnostics is None else diagnostics()

    def prepare_solve(self, context: SolvePreparationContext) -> None:
        """Freeze material references at the start of a fixed-basis solve."""

        self.capture_stage_references(list(context.metric_results))
        self.last_sensitivity_audit = (
            None
            if self.sensitivity_audit is None
            else self.sensitivity_audit.prepare(context)
        )

    def set_references(
        self, references: dict[str, MaterialFeatureReference]
    ) -> None:
        self.references = dict(references)
        self._validate_references(require_all=True)

    def capture_stage_references(
        self,
        metric_results: list[MetricResult],
        *,
        noise_floors: dict[str, float] | None = None,
    ) -> dict[str, MaterialFeatureReference]:
        floors = noise_floors or {
            name: reference.noise_floor for name, reference in self.references.items()
        }
        captured: dict[str, MaterialFeatureReference] = {}
        for term in self.feature_terms:
            feature = self._evaluate_term(term, metric_results)
            floor = float(floors.get(term.name, 0.0))
            reference = max(
                feature.value,
                floor + max(np.finfo(np.float64).eps, abs(floor) * 1.0e-12),
            )
            captured[term.name] = MaterialFeatureReference(floor, reference)
        self.set_references(captured)
        return dict(captured)

    def evaluate(self, metric_results: list[MetricResult]) -> float:
        global_cost = 0.0
        if self.alpha < 1.0:
            assert self.global_objective is not None
            global_cost = float(self.global_objective.evaluate(metric_results))
        diagnostics: list[MaterialInformationFeatureDiagnostic] = []
        weighted_values: list[float] = []
        if self.alpha > 0.0:
            self._validate_references(require_all=True)
            for term in self.feature_terms:
                raw = self._evaluate_term(term, metric_results)
                reference = self.references[term.name]
                denominator = reference.stage_reference - reference.noise_floor
                scaled_excess = (
                    raw.value - reference.noise_floor
                ) / denominator
                normalised = float(smooth_positive_part(
                    scaled_excess,
                    temperature=self.positive_part_temperature,
                ))
                weighted = float(term.weight * normalised)
                weighted_values.append(weighted)
                diagnostics.append(
                    MaterialInformationFeatureDiagnostic(
                        name=term.name,
                        raw_value=raw.value,
                        noise_floor=reference.noise_floor,
                        stage_reference=reference.stage_reference,
                        normalised_value=normalised,
                        weighted_value=weighted,
                        valid_count=raw.valid_count,
                        effective_count=raw.effective_count,
                    )
                )
        material_smooth_max = _smooth_max(
            weighted_values, self.smooth_max_temperature
        ) if weighted_values else 0.0
        material_mean = float(np.mean(weighted_values)) if weighted_values else 0.0
        material_cost = (
            (1.0 - self.mean_fraction) * material_smooth_max
            + self.mean_fraction * material_mean
        )
        total = (1.0 - self.alpha) * global_cost + self.alpha * material_cost
        self.last_result = MaterialInformationObjectiveResult(
            total_cost=float(total),
            global_cost=global_cost,
            material_cost=material_cost,
            material_smooth_max=material_smooth_max,
            material_mean=material_mean,
            features=tuple(diagnostics),
        )
        return float(total)

    def diagnostics(self) -> dict[str, object]:
        result: dict[str, object] = {
            "type": type(self).__name__,
            "alpha": self.alpha,
            "smooth_max_temperature": self.smooth_max_temperature,
            "mean_fraction": self.mean_fraction,
            "positive_part_temperature": self.positive_part_temperature,
            "references": {
                name: asdict(reference) for name, reference in self.references.items()
            },
        }
        if self.last_result is not None:
            result["last_result"] = asdict(self.last_result)
        if self.last_sensitivity_audit is not None:
            result["sensitivity_audit"] = (
                self.last_sensitivity_audit.diagnostics()
            )
        return result

    def _evaluate_term(
        self, term: MaterialFeatureTerm, metric_results: list[MetricResult]
    ) -> ResidualFeatureResult:
        try:
            metric_result = metric_results[term.metric_result_index]
        except IndexError as exc:
            raise ValueError(
                f"Feature {term.name!r} selects absent metric result "
                f"{term.metric_result_index}."
            ) from exc
        metadata = metric_result.additional_fields or {}
        residual = metadata.get("normalised_gap", metadata.get("normalised_residual", metric_result.residual))
        if residual is None:
            raise ValueError(f"Feature {term.name!r} metric has no residual array.")
        values = np.asarray(residual, dtype=np.float64)
        if term.frame_indices is not None:
            values = values[np.asarray(term.frame_indices, dtype=np.int64)]
        weights = _observation_weights(metadata, values, term.frame_indices)
        match term.reduction:
            case MaterialFeatureReduction.RMS:
                return weighted_rms(values, weights)
            case MaterialFeatureReduction.CVAR_ABS:
                return weighted_cvar_abs(values, weights, quantile=term.quantile)
            case MaterialFeatureReduction.COHERENT_RMS:
                return coherent_rms(
                    values,
                    weights,
                    sigma_pixels=term.sigma_pixels,
                    spatial_axes=term.spatial_axes,
                )
            case MaterialFeatureReduction.PROJECTED_RMS:
                if term.projection_basis is None:
                    raise ValueError(f"Feature {term.name!r} requires a projection basis.")
                return projected_rms(values, term.projection_basis, weights)
        raise AssertionError(f"Unhandled reduction {term.reduction!r}.")

    def _validate_references(self, *, require_all: bool) -> None:
        configured = {term.name for term in self.feature_terms}
        unknown = set(self.references) - configured
        if unknown:
            raise ValueError(f"References configured for unknown features: {sorted(unknown)}")
        if require_all and set(self.references) != configured:
            missing = configured - set(self.references)
            raise ValueError(f"Missing references for features: {sorted(missing)}")


def _observation_weights(
    metadata: dict,
    values: npt.NDArray[np.float64],
    frame_indices: tuple[int, ...] | None,
) -> npt.NDArray[np.float64]:
    weights = np.ones(values.shape, dtype=np.float64)
    temporal = metadata.get("temporal_weights")
    if temporal is not None and values.ndim >= 1:
        temporal_array = np.asarray(temporal, dtype=np.float64)
        if frame_indices is not None:
            temporal_array = temporal_array[np.asarray(frame_indices, dtype=np.int64)]
        shape = (temporal_array.size,) + (1,) * (values.ndim - 1)
        weights *= temporal_array.reshape(shape)
    spatial = metadata.get("spatial_weights")
    if spatial is not None and values.ndim >= 2:
        spatial_array = np.asarray(spatial, dtype=np.float64)
        try:
            weights *= np.broadcast_to(spatial_array, values.shape)
        except ValueError:
            shape = (1,) * (values.ndim - spatial_array.ndim) + spatial_array.shape
            weights *= np.broadcast_to(spatial_array.reshape(shape), values.shape)
    return weights


def _smooth_max(values: Sequence[float], temperature: float) -> float:
    resolved = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(resolved))
    return float(maximum + temperature * np.log(np.mean(np.exp((resolved - maximum) / temperature))))
