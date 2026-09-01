"""Hard-guarded, sensitivity-gated fine+broad EGI primary objective."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import threading
import time
from typing import Callable

import numpy as np
import numpy.typing as npt

from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.measurementnoise import (
    MeasurementNoiseFloorConfig,
    MeasurementNoiseMode,
    measurement_noise_realisation,
)
from pyvale.vfm.metric import IMetric, MetricResult
from pyvale.vfm.metricequilibriumgap import EquilibriumGapMetric
from pyvale.vfm.metricsliceforce import SliceWiseForceReconstructionMetric
from pyvale.vfm.objectivefunc import IScalarObjectiveFunction
from pyvale.vfm.objectivefuncsensitivitygated import (
    SensitivityGatedEgiObjective,
    SensitivityGatedObjectiveConfig,
)
from pyvale.vfm.residualblocks import (
    CanonicalResidualLayout,
    ResidualBlockSpec,
    prepare_canonical_residual_layout,
)
from pyvale.vfm.loadregimes import resolve_load_regimes
from pyvale.vfm.solvepreparation import SolvePreparationContext
from pyvale.vfm.spatialparam import ISpatialParameterisation


FloatArray = npt.NDArray[np.float64]


@dataclass(slots=True, frozen=True)
class GuardReference:
    """Frozen parent/noise reference for one hard guard."""

    parent: float
    noise_floor: float
    reference: float
    limit: float


@dataclass(slots=True, frozen=True)
class GuardedEgiPrimaryConfig:
    """Configuration for the additive ``guarded_egi_primary`` mode."""

    fine_noise_scale: float
    broad_noise_scale: float
    measurement_noise: MeasurementNoiseFloorConfig = MeasurementNoiseFloorConfig()
    guard_relaxation: float = 0.10
    parameter_names: tuple[str, ...] = ("yield_strength", "hardening_modulus")
    perturbation_factor: float = 0.01
    sensitivity_scaling_percentile: float = 95.0
    gate_start: float = 0.05
    gate_full: float = 0.30
    gate_start_quantile: float | None = 0.0
    gate_full_quantile: float | None = 0.90
    positive_activity_floor: float = 1.0e-6
    run_identifier: str = ""
    candidate_log_path: str | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.fine_noise_scale) or self.fine_noise_scale <= 0.0:
            raise ValueError("fine_noise_scale must be finite and positive.")
        if not np.isfinite(self.broad_noise_scale) or self.broad_noise_scale <= 0.0:
            raise ValueError("broad_noise_scale must be finite and positive.")
        if not np.isclose(self.guard_relaxation, 0.10, rtol=0.0, atol=1.0e-15):
            raise ValueError("guard_relaxation is provisionally fixed at 0.10.")


@dataclass(slots=True, frozen=True)
class GuardedEgiPrimaryResult:
    total_cost: float
    fine_gated_cost: float | None
    broad_gated_cost: float | None
    fre_guard_value: float
    broad_unmasked_guard_value: float | None
    feasible: bool
    rejection_reason: str


class CandidateAuditRecorder:
    """Thread-safe JSONL sink and measured per-solve short-circuit counters."""

    def __init__(self, path: str | None) -> None:
        self.path = None if path is None else Path(path)
        self._lock = threading.Lock()
        self._stage = 0
        self._next_index = 0
        self._records: list[dict[str, object]] = []

    def __deepcopy__(self, memo):
        # A generic objective copy (for example an optional basis-placement
        # screening solve) must not append into the active production solve's
        # audit stream. Pattern-search worker copies opt in explicitly below.
        return CandidateAuditRecorder(None)

    def start_solve(self, stage: int) -> None:
        with self._lock:
            self._stage = int(stage)
            self._next_index = 0
            self._records = []

    def append(self, record: dict[str, object]) -> None:
        with self._lock:
            payload = dict(record)
            payload["bf_stage"] = self._stage
            payload["evaluation_index"] = self._next_index
            self._next_index += 1
            self._records.append(payload)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def summary(self) -> dict[str, object]:
        with self._lock:
            records = list(self._records)
        total = len(records)
        fre_rejected = sum(row["rejection_reason"] == "FRE" for row in records)
        broad_rejected = sum(row["rejection_reason"] == "BROAD" for row in records)
        feasible = sum(row["rejection_reason"] == "NONE" for row in records)
        reached_broad = total - fre_rejected
        reached_fine = feasible
        fraction = lambda count: 0.0 if total == 0 else count / total
        return {
            "total_candidate_evaluations": total,
            "rejected_at_fre": fre_rejected,
            "rejected_at_fre_fraction": fraction(fre_rejected),
            "rejected_at_broad": broad_rejected,
            "rejected_at_broad_fraction": fraction(broad_rejected),
            "reaching_fine_egi": reached_fine,
            "reaching_fine_egi_fraction": fraction(reached_fine),
            "fully_feasible": feasible,
            "fully_feasible_fraction": fraction(feasible),
            "total_stress_time_seconds": float(sum(float(row["stress_reconstruction_time_seconds"]) for row in records)),
            "total_fre_time_seconds": float(sum(float(row["fre_evaluation_time_seconds"]) for row in records)),
            "total_broad_egi_time_seconds": float(sum(float(row["broad_egi_evaluation_time_seconds"] or 0.0) for row in records)),
            "total_fine_egi_time_seconds": float(sum(float(row["fine_egi_evaluation_time_seconds"] or 0.0) for row in records)),
            "broad_egi_evaluations_avoided": fre_rejected,
            "fine_egi_evaluations_avoided": fre_rejected + broad_rejected,
            "broad_egi_evaluations_reached": reached_broad,
        }


class GuardedEgiPrimaryObjective(IScalarObjectiveFunction):
    """Fine+broad gated EGI mean behind frozen FRE and broad hard guards.

    Metric order is deliberately fixed to ``[FRE, fine EGI, broad EGI]``.
    Candidate stress is reconstructed by the shared optimiser path once. This
    class then evaluates FRE, broad EGI and fine EGI lazily in that order.
    """

    requires_explicit_parent_state = True

    def __init__(
        self,
        config: GuardedEgiPrimaryConfig,
        *,
        diagnostic_callback: Callable[[str, dict[str, object]], None] | None = None,
        basis_growth_objective: IScalarObjectiveFunction | None = None,
    ) -> None:
        self.config = config
        self.diagnostic_callback = diagnostic_callback
        self.global_objective = basis_growth_objective
        self.last_result: GuardedEgiPrimaryResult | None = None
        self._gate_delegate = SensitivityGatedEgiObjective(
            SensitivityGatedObjectiveConfig(
                parameter_names=config.parameter_names,
                perturbation_factor=config.perturbation_factor,
                sensitivity_scaling_percentile=config.sensitivity_scaling_percentile,
                gate_start=config.gate_start,
                gate_full=config.gate_full,
                gate_start_quantile=config.gate_start_quantile,
                gate_full_quantile=config.gate_full_quantile,
                positive_activity_floor=config.positive_activity_floor,
                egi_roles=("fine", "broad"),
                egi_noise_scales=(config.fine_noise_scale, config.broad_noise_scale),
                force_noise_scale=1.0,
                refresh_every_solves=None,
            )
        )
        self._fre_layout: CanonicalResidualLayout | None = None
        self._fine_layout: CanonicalResidualLayout | None = None
        self._broad_gated_layout: CanonicalResidualLayout | None = None
        self._broad_guard_layout: CanonicalResidualLayout | None = None
        self._fre_reference: GuardReference | None = None
        self._broad_reference: GuardReference | None = None
        self._prepared_solve: int | None = None
        self._diagnostics: dict[str, object] = {}
        self._last_fre_diagnostics: dict[str, object] | None = None
        self._recorder = CandidateAuditRecorder(config.candidate_log_path)

    @property
    def baseline(self):
        return getattr(self.global_objective, "baseline", None)

    @property
    def spatial_weighting(self):
        return None

    def resolve_from_prior_phase(self, metric_results) -> None:
        resolver = getattr(self.global_objective, "resolve_from_prior_phase", None)
        if resolver is not None:
            resolver(metric_results)

    def baseline_diagnostics(self) -> dict[str, object]:
        diagnostics = getattr(self.global_objective, "baseline_diagnostics", None)
        return {} if diagnostics is None else diagnostics()

    def spatial_weighting_diagnostics(self) -> dict[str, object]:
        return {"enabled": False, "mode": "frozen_two_parameter_sensitivity_gate"}

    def clone_for_candidate_evaluation(self) -> GuardedEgiPrimaryObjective:
        """Clone mutable diagnostics while sharing the active audit recorder."""

        clone = copy.deepcopy(self)
        clone._recorder = self._recorder
        return clone

    def prepare_solve(self, context: SolvePreparationContext) -> dict[str, object]:
        self._validate_metrics(context.metrics)
        gate_diagnostics = self._gate_delegate.prepare_solve(context)
        gate = self._gate_delegate._weights
        if gate is None:
            raise RuntimeError("Sensitivity gate preparation produced no weights.")
        parent_results = context.parent_metric_results
        if len(parent_results) != 3:
            raise ValueError("guarded_egi_primary requires parent results ordered FRE, fine, broad.")
        timestep_count = context.experiment_data.strain.shape[0]
        regimes = resolve_load_regimes(np.zeros(timestep_count))
        fre_temporal = _temporal_weights(parent_results[0], timestep_count)
        fine_temporal = _temporal_weights(parent_results[1], timestep_count)
        broad_temporal = _temporal_weights(parent_results[2], timestep_count)
        self._fre_layout = prepare_canonical_residual_layout(
            [parent_results[0]], regimes,
            [ResidualBlockSpec("fre", 0, "all", "fre", residual_field="normalised_residual", observation_weights=fre_temporal[:, None])],
        )
        self._fine_layout = prepare_canonical_residual_layout(
            [parent_results[1]], regimes,
            [ResidualBlockSpec("fine", 0, "all", "egi", residual_field="normalised_gap", noise_scale=self.config.fine_noise_scale, observation_weights=gate * fine_temporal[:, None, None])],
        )
        self._broad_gated_layout = prepare_canonical_residual_layout(
            [parent_results[2]], regimes,
            [ResidualBlockSpec("broad_gated", 0, "all", "egi", residual_field="normalised_gap", noise_scale=self.config.broad_noise_scale, observation_weights=gate * broad_temporal[:, None, None])],
        )
        self._broad_guard_layout = prepare_canonical_residual_layout(
            [parent_results[2]], regimes,
            [ResidualBlockSpec("broad_guard", 0, "all", "egi", residual_field="normalised_gap", observation_weights=broad_temporal[:, None, None])],
        )

        fre_parent = _layout_rms(self._fre_layout, parent_results[0])
        broad_parent = _layout_rms(self._broad_guard_layout, parent_results[2])
        fre_floor, broad_floor, floor_diagnostics = self._measurement_noise_floors(
            context,
            parent_results,
        )
        self._fre_reference = _guard_reference(
            fre_parent, fre_floor, self.config.guard_relaxation
        )
        self._broad_reference = _guard_reference(
            broad_parent, broad_floor, self.config.guard_relaxation
        )
        self._prepared_solve = context.solve_iteration
        self._last_fre_diagnostics = None
        self._recorder.start_solve(context.solve_iteration + 1)
        self._diagnostics = {
            "mode": "guarded_egi_primary",
            "bf_stage": context.solve_iteration + 1,
            "primary": {
                "definition": "equal mean of separately noise-normalised gated fine and broad EGI RMS",
                "fine_noise_scale": self.config.fine_noise_scale,
                "broad_noise_scale": self.config.broad_noise_scale,
                "middle_egi_used": False,
                "fre_in_scalar": False,
                "unmasked_broad_in_scalar": False,
            },
            "guard_relaxation": self.config.guard_relaxation,
            "fre_guard": asdict(self._fre_reference),
            "broad_unmasked_guard": asdict(self._broad_reference),
            "measurement_noise_floor": floor_diagnostics,
            "sensitivity_gate": gate_diagnostics,
        }
        if self.diagnostic_callback is not None:
            self.diagnostic_callback("guarded_egi_preparation", self._diagnostics)
        return copy.deepcopy(self._diagnostics)

    def _measurement_noise_floors(
        self,
        context: SolvePreparationContext,
        parent_results: tuple[MetricResult, ...],
    ) -> tuple[float, float, dict[str, object]]:
        config = self.config.measurement_noise
        metadata = config.metadata()
        if config.mode is MeasurementNoiseMode.PARENT_ONLY:
            metadata.update({
                "fre_realisation_rms": [],
                "broad_realisation_rms": [],
                "fre_noise_floor": 0.0,
                "broad_noise_floor": 0.0,
            })
            return 0.0, 0.0, metadata
        fre_metric = context.metrics[0]
        broad_metric = context.metrics[2]
        assert isinstance(fre_metric, SliceWiseForceReconstructionMetric)
        assert isinstance(broad_metric, EquilibriumGapMetric)
        axis = fre_metric.slice_partition.axis if fre_metric.slice_partition is not None else fre_metric.slice_config.axis
        fre_values: list[float] = []
        broad_values: list[float] = []
        parent_fre = np.asarray(parent_results[0].additional_fields["normalised_residual"], dtype=np.float64)
        parent_broad = np.asarray(parent_results[2].additional_fields["normalised_gap"], dtype=np.float64)
        for seed in config.seeds:
            noisy_experiment = measurement_noise_realisation(
                context.experiment_data, config, seed, force_axis=axis,
            )
            noisy_stress = context.constitutive_law.calculate_stress(
                noisy_experiment.strain,
                context.parent_parameter_maps,
            )
            noisy_fre_metric = copy.deepcopy(fre_metric)
            noisy_broad_metric = copy.deepcopy(broad_metric)
            noisy_fre_metric.initialise(noisy_experiment)
            noisy_broad_metric.initialise(noisy_experiment)
            noisy_fre = noisy_fre_metric.evaluate_force_recon_error(
                noisy_stress, noisy_experiment
            ).metric_result
            noisy_broad = noisy_broad_metric.evaluate_equilibrium_gap(
                noisy_stress, include_diagnostics=True
            ).metric_result
            fre_delta = np.asarray(noisy_fre.additional_fields["normalised_residual"], dtype=np.float64) - parent_fre
            broad_delta = np.asarray(noisy_broad.additional_fields["normalised_gap"], dtype=np.float64) - parent_broad
            fre_values.append(_layout_rms(
                self._require_layout(self._fre_layout),
                MetricResult(additional_fields={"normalised_residual": fre_delta}),
            ))
            broad_values.append(_layout_rms(
                self._require_layout(self._broad_guard_layout),
                MetricResult(additional_fields={"normalised_gap": broad_delta}),
            ))
        fre_floor = float(np.quantile(fre_values, config.quantile))
        broad_floor = float(np.quantile(broad_values, config.quantile))
        metadata.update({
            "fre_realisation_rms": fre_values,
            "broad_realisation_rms": broad_values,
            "fre_noise_floor": fre_floor,
            "broad_noise_floor": broad_floor,
            "floor_definition": "empirical Q95 of RMS(r_noisy - r_parent)",
        })
        return fre_floor, broad_floor, metadata

    def evaluate_candidate_stress(
        self,
        stress: FloatArray,
        constitutive_law,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
        metrics: list[IMetric],
        experiment_data: ExperimentData,
        *,
        stress_reconstruction_time_seconds: float,
    ) -> float:
        self._ensure_prepared()
        self._validate_metrics(tuple(metrics))
        fre_reference = self._require_reference(self._fre_reference)
        broad_reference = self._require_reference(self._broad_reference)
        record = self._base_record(stress_reconstruction_time_seconds)

        started = time.perf_counter()
        fre_result = metrics[0].evaluate(
            stress, constitutive_law, parameter_map_size,
            spatial_parameterisations, experiment_data,
        )
        record["fre_evaluation_time_seconds"] = time.perf_counter() - started
        fre_value = _layout_rms(self._require_layout(self._fre_layout), fre_result)
        self._last_fre_diagnostics = _accepted_fre_diagnostics(
            fre_result,
            fre_value=fre_value,
        )
        record["fre_value"] = fre_value
        record["fre_passed"] = _passes_limit(fre_value, fre_reference.limit)
        if not record["fre_passed"]:
            record["rejection_reason"] = "FRE"
            self.last_result = GuardedEgiPrimaryResult(
                np.inf, None, None, fre_value, None, False, "FRE"
            )
            self._recorder.append(record)
            return np.inf

        started = time.perf_counter()
        assert isinstance(metrics[2], EquilibriumGapMetric)
        broad_result = metrics[2].evaluate_equilibrium_gap(
            stress, include_diagnostics=True
        ).metric_result
        record["broad_egi_evaluation_time_seconds"] = time.perf_counter() - started
        broad_guard = _layout_rms(
            self._require_layout(self._broad_guard_layout), broad_result
        )
        broad_gated = _layout_rms(
            self._require_layout(self._broad_gated_layout), broad_result
        )
        record["broad_unmasked_value"] = broad_guard
        record["gated_broad_value"] = broad_gated
        record["broad_passed"] = _passes_limit(broad_guard, broad_reference.limit)
        if not record["broad_passed"]:
            record["rejection_reason"] = "BROAD"
            self.last_result = GuardedEgiPrimaryResult(
                np.inf, None, broad_gated, fre_value, broad_guard, False, "BROAD"
            )
            self._recorder.append(record)
            return np.inf

        started = time.perf_counter()
        assert isinstance(metrics[1], EquilibriumGapMetric)
        fine_result = metrics[1].evaluate_equilibrium_gap(
            stress, include_diagnostics=True
        ).metric_result
        record["fine_egi_evaluation_time_seconds"] = time.perf_counter() - started
        fine_gated = _layout_rms(
            self._require_layout(self._fine_layout), fine_result
        )
        primary = equal_mean_gated_egi_primary(fine_gated, broad_gated)
        record["gated_fine_value"] = fine_gated
        record["j_primary"] = primary
        record["rejection_reason"] = "NONE"
        self.last_result = GuardedEgiPrimaryResult(
            primary, fine_gated, broad_gated, fre_value, broad_guard, True, "NONE"
        )
        self._recorder.append(record)
        return primary

    def evaluate(self, metric_results: list[MetricResult]) -> float:
        """Evaluate already-computed results; optimiser candidates use lazy path."""
        self._ensure_prepared()
        if len(metric_results) != 3:
            raise ValueError("guarded_egi_primary expects FRE, fine EGI and broad EGI results.")
        fre = _layout_rms(self._require_layout(self._fre_layout), metric_results[0])
        broad_guard = _layout_rms(self._require_layout(self._broad_guard_layout), metric_results[2])
        broad_gated = _layout_rms(self._require_layout(self._broad_gated_layout), metric_results[2])
        fine_gated = _layout_rms(self._require_layout(self._fine_layout), metric_results[1])
        fre_pass = _passes_limit(fre, self._require_reference(self._fre_reference).limit)
        broad_pass = _passes_limit(broad_guard, self._require_reference(self._broad_reference).limit)
        if not fre_pass or not broad_pass:
            reason = "FRE" if not fre_pass else "BROAD"
            self.last_result = GuardedEgiPrimaryResult(
                np.inf, fine_gated, broad_gated, fre, broad_guard, False, reason
            )
            return np.inf
        primary = equal_mean_gated_egi_primary(fine_gated, broad_gated)
        self.last_result = GuardedEgiPrimaryResult(
            primary, fine_gated, broad_gated, fre, broad_guard, True, "NONE"
        )
        return primary

    def diagnostics(self) -> dict[str, object]:
        result = copy.deepcopy(self._diagnostics)
        result["candidate_summary"] = self._recorder.summary()
        if self._last_fre_diagnostics is not None:
            result["accepted_fre"] = copy.deepcopy(self._last_fre_diagnostics)
        if self.last_result is not None:
            result["last_result"] = asdict(self.last_result)
        return result

    def finalize_solve(self) -> dict[str, object]:
        summary = self._recorder.summary()
        if self._last_fre_diagnostics is not None:
            summary["accepted_fre"] = copy.deepcopy(self._last_fre_diagnostics)
        if self.diagnostic_callback is not None:
            self.diagnostic_callback("guarded_egi_solve_summary", summary)
        return summary

    def _base_record(self, stress_seconds: float) -> dict[str, object]:
        fre = self._require_reference(self._fre_reference)
        broad = self._require_reference(self._broad_reference)
        return {
            "run_identifier": self.config.run_identifier,
            "stress_reconstruction_time_seconds": stress_seconds,
            "fre_evaluation_time_seconds": 0.0,
            "fre_value": None,
            "fre_parent": fre.parent,
            "fre_noise_floor": fre.noise_floor,
            "fre_reference": fre.reference,
            "fre_limit": fre.limit,
            "fre_passed": None,
            "broad_egi_evaluation_time_seconds": None,
            "broad_unmasked_value": None,
            "broad_parent": broad.parent,
            "broad_noise_floor": broad.noise_floor,
            "broad_reference": broad.reference,
            "broad_limit": broad.limit,
            "broad_passed": None,
            "gated_broad_value": None,
            "fine_egi_evaluation_time_seconds": None,
            "gated_fine_value": None,
            "j_primary": None,
            "rejection_reason": None,
        }

    def _validate_metrics(self, metrics: tuple[IMetric, ...]) -> None:
        if len(metrics) != 3:
            raise ValueError(
                "guarded_egi_primary requires exactly three metrics: FRE, fine EGI, broad EGI; middle EGI is forbidden."
            )
        if not isinstance(metrics[0], SliceWiseForceReconstructionMetric):
            raise TypeError("Metric 0 must be SliceWiseForceReconstructionMetric.")
        if not isinstance(metrics[1], EquilibriumGapMetric) or not isinstance(metrics[2], EquilibriumGapMetric):
            raise TypeError("Metrics 1 and 2 must be fine and broad EquilibriumGapMetric instances.")
        fine_area = int(np.prod(metrics[1].window_size))
        broad_area = int(np.prod(metrics[2].window_size))
        if fine_area >= broad_area:
            raise ValueError("Fine EGI support must be smaller than broad EGI support.")

    def _ensure_prepared(self) -> None:
        if self._prepared_solve is None:
            raise RuntimeError("guarded_egi_primary must be prepared before evaluation.")

    @staticmethod
    def _require_layout(layout: CanonicalResidualLayout | None) -> CanonicalResidualLayout:
        if layout is None:
            raise RuntimeError("Guarded objective residual layout is not prepared.")
        return layout

    @staticmethod
    def _require_reference(reference: GuardReference | None) -> GuardReference:
        if reference is None:
            raise RuntimeError("Guarded objective reference is not prepared.")
        return reference


def _layout_rms(layout: CanonicalResidualLayout, result: MetricResult) -> float:
    vector = layout.evaluate([result])
    return float(np.linalg.norm(vector.weighted))


def _accepted_fre_diagnostics(
    result: MetricResult,
    *,
    fre_value: float,
) -> dict[str, object]:
    """Return durable accepted-state FRE profiles and scalar summaries."""

    fields = result.additional_fields or {}
    relative = np.asarray(fields["normalised_residual"], dtype=np.float64)
    temporal_weights = np.asarray(fields["temporal_weights"], dtype=np.float64)
    weighted_profile = 100.0 * np.sqrt(
        np.nansum(temporal_weights[:, None] * relative**2, axis=0)
    )
    finite_percent = 100.0 * relative[np.isfinite(relative)]
    diagnostics: dict[str, object] = {
        "relative_fre_percent": 100.0 * relative.copy(),
        "weighted_temporal_rms_percent_by_slice": weighted_profile,
        "weighted_spatiotemporal_rms_percent": 100.0 * float(fre_value),
        "p95_absolute_fre_percent": (
            float("nan")
            if finite_percent.size == 0
            else float(np.percentile(np.abs(finite_percent), 95.0))
        ),
        "maximum_absolute_fre_percent": (
            float("nan")
            if finite_percent.size == 0
            else float(np.max(np.abs(finite_percent)))
        ),
    }
    if "reconstructed_force" in fields:
        diagnostics["reconstructed_force_n"] = np.asarray(
            fields["reconstructed_force"], dtype=np.float64
        ).copy()
    if "applied_longitudinal_force" in fields:
        diagnostics["applied_force_n"] = np.asarray(
            fields["applied_longitudinal_force"], dtype=np.float64
        ).copy()
    for name in (
        "force_integration_domain_correction_enabled",
        "force_integration_scale_factors",
        "force_integration_measured_areas",
        "force_integration_target_areas",
        "force_integration_measured_widths",
        "force_integration_target_widths",
        "force_integration_represented_fractions",
    ):
        if name in fields:
            value = fields[name]
            diagnostics[name] = (
                np.asarray(value).copy()
                if isinstance(value, np.ndarray)
                else value
            )
    return diagnostics


def _temporal_weights(result: MetricResult, count: int) -> FloatArray:
    fields = result.additional_fields or {}
    raw = fields.get("temporal_weights")
    if raw is None:
        return np.ones(count, dtype=np.float64)
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != (count,):
        raise ValueError(f"Temporal weights shape {values.shape} does not match {(count,)}.")
    return np.where(np.isfinite(values) & (values > 0.0), values, 0.0)


def _guard_reference(parent: float, noise_floor: float, relaxation: float) -> GuardReference:
    if any(not np.isfinite(value) or value < 0.0 for value in (parent, noise_floor)):
        raise ValueError("Guard parent and noise floor must be finite and non-negative.")
    reference = max(float(parent), float(noise_floor))
    return GuardReference(float(parent), float(noise_floor), reference, (1.0 + relaxation) * reference)


def _passes_limit(value: float, limit: float) -> bool:
    tolerance = 1.0e-12 * max(1.0, abs(limit))
    return bool(np.isfinite(value) and value <= limit + tolerance)


def equal_mean_gated_egi_primary(
    fine_noise_normalised_rms: float,
    broad_noise_normalised_rms: float,
) -> float:
    """Return the prescribed equal mean of the two gated EGI blocks."""

    values = np.asarray(
        (fine_noise_normalised_rms, broad_noise_normalised_rms),
        dtype=np.float64,
    )
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Gated EGI block RMS values must be finite and non-negative.")
    return float(np.mean(values))
