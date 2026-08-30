"""Conventional EGI/FRE objective with frozen two-perturbation observation gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from pyvale.vfm.metricsbvf import calculate_parameter_stress_sensitivities
from pyvale.vfm.loadregimes import resolve_load_regimes
from pyvale.vfm.objectivefunc import IScalarObjectiveFunction
from pyvale.vfm.residualblocks import CanonicalResidualLayout, ResidualBlockSpec
from pyvale.vfm.solvepreparation import SolvePreparationContext


FloatArray = npt.NDArray[np.float64]


@dataclass(slots=True, frozen=True)
class SensitivityGatedObjectiveConfig:
    """Small set of transparent controls for the minimalist objective."""

    parameter_names: tuple[str, ...] = ("yield_strength", "hardening_modulus")
    perturbation_factor: float = 0.01
    sensitivity_scaling_percentile: float = 95.0
    gate_start: float = 0.05
    gate_full: float = 0.30
    gate_start_quantile: float | None = None
    gate_full_quantile: float | None = None
    positive_activity_floor: float = 1.0e-6
    egi_noise_scales: tuple[float, float, float] = (1.0, 1.0, 1.0)
    force_noise_scale: float = 1.0
    force_weight: float = 0.15
    broad_guard_weight: float = 0.10
    refresh_every_solves: int | None = None

    def __post_init__(self) -> None:
        if not self.parameter_names or len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("parameter_names must be non-empty and unique.")
        if not 0.0 < self.perturbation_factor < 1.0:
            raise ValueError("perturbation_factor must lie in (0, 1).")
        if not 0.0 < self.sensitivity_scaling_percentile <= 100.0:
            raise ValueError("sensitivity_scaling_percentile must lie in (0, 100].")
        if not 0.0 <= self.gate_start < self.gate_full <= 1.0:
            raise ValueError("Require 0 <= gate_start < gate_full <= 1.")
        if (self.gate_start_quantile is None) != (self.gate_full_quantile is None):
            raise ValueError("Provide both gate quantiles or neither.")
        if self.gate_start_quantile is not None and not (
            0.0 <= self.gate_start_quantile < self.gate_full_quantile <= 1.0
        ):
            raise ValueError("Require 0 <= gate_start_quantile < gate_full_quantile <= 1.")
        if not np.isfinite(self.positive_activity_floor) or not 0.0 <= self.positive_activity_floor < 1.0:
            raise ValueError("positive_activity_floor must lie in [0, 1).")
        if len(self.egi_noise_scales) != 3 or any(
            not np.isfinite(value) or value <= 0.0 for value in self.egi_noise_scales
        ):
            raise ValueError("egi_noise_scales must contain three positive values.")
        if not np.isfinite(self.force_noise_scale) or self.force_noise_scale <= 0.0:
            raise ValueError("force_noise_scale must be positive.")
        if self.force_weight < 0.0 or self.broad_guard_weight < 0.0 or (
            self.force_weight + self.broad_guard_weight >= 1.0
        ):
            raise ValueError("Guard weights must be non-negative and sum to less than one.")
        if self.refresh_every_solves is not None and self.refresh_every_solves < 1:
            raise ValueError("refresh_every_solves must be positive when supplied.")


@dataclass(slots=True, frozen=True)
class SensitivityGatedObjectiveResult:
    total_cost: float
    informative_egi_cost: float
    fine_cost: float
    middle_cost: float
    broad_cost: float
    force_guard_cost: float
    broad_guard_cost: float


class SensitivityGatedEgiObjective(IScalarObjectiveFunction):
    """Noise-normalised EGI RMS with frozen stress-sensitivity soft gates.

    One global perturbation is evaluated for each configured material
    parameter. Their pointwise stress-response magnitudes create one common
    space-time gate used by fine, middle and broad EGI. FRE and the duplicate
    full broad-EGI guard remain unmasked. No optimiser-DOF sensitivity, SVD,
    Fisher matrix or projected residual is constructed.
    """

    def __init__(
        self,
        config: SensitivityGatedObjectiveConfig,
        *,
        diagnostic_callback: Callable[[str, dict[str, object]], None] | None = None,
        basis_growth_objective: IScalarObjectiveFunction | None = None,
    ) -> None:
        self.config = config
        self.diagnostic_callback = diagnostic_callback
        self.global_objective = basis_growth_objective
        self._layout: CanonicalResidualLayout | None = None
        self._weights: FloatArray | None = None
        self._prepared_at_solve: int | None = None
        self._diagnostics: dict[str, object] = {}
        self.last_result: SensitivityGatedObjectiveResult | None = None

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
        return {"enabled": False, "mode": "objective_internal_space_time_gate"}

    def prepare_solve(self, context: SolvePreparationContext) -> dict[str, object]:
        refresh = self._layout is None or (
            self.config.refresh_every_solves is not None
            and context.solve_iteration % self.config.refresh_every_solves == 0
        )
        if not refresh:
            return {**self._diagnostics, "refreshed": False}

        missing = [name for name in self.config.parameter_names if name not in context.parameter_maps]
        if missing:
            raise KeyError(f"Sensitivity-gate parameters are unavailable: {missing}.")
        sensitivities = calculate_parameter_stress_sensitivities(
            context.experiment_data.strain,
            context.stress,
            context.constitutive_law,
            context.parameter_maps,
            list(self.config.parameter_names),
            perturbation_factor=self.config.perturbation_factor,
        )
        activity = {}
        for name, item in sensitivities.items():
            values = np.sqrt(np.nansum(item.total**2, axis=1))
            values[~np.any(np.isfinite(item.total), axis=1)] = np.nan
            activity[name] = values
        scaled = {
            name: _normalise_activity(values, self.config.sensitivity_scaling_percentile)
            for name, values in activity.items()
        }
        combined = np.maximum.reduce(list(scaled.values()))
        combined[np.isfinite(combined) & (combined < self.config.positive_activity_floor)] = 0.0
        gate_start, gate_full = self._resolve_gate_thresholds(combined)
        weights = _smooth_gate(combined, gate_start, gate_full)
        weights[~np.isfinite(combined)] = np.nan
        if not np.any(np.isfinite(weights) & (weights > 0.0)):
            raise ValueError("Sensitivity gate retained no informative observations.")

        specs = [
            ResidualBlockSpec(
                "fre_guard", 0, "all", "fre", role="fre_guard",
                residual_field="normalised_residual",
                noise_scale=self.config.force_noise_scale,
            )
        ]
        for index, role in enumerate(("fine", "middle", "broad"), start=1):
            specs.append(ResidualBlockSpec(
                f"egi_{role}_informative", index, "all", "egi",
                role="training", residual_field="normalised_gap",
                noise_scale=self.config.egi_noise_scales[index - 1],
                observation_weights=weights,
            ))
        specs.append(ResidualBlockSpec(
            "egi_broad_guard", 3, "all", "egi", role="broad_egi_guard",
            residual_field="normalised_gap",
            noise_scale=self.config.egi_noise_scales[2],
        ))
        self._layout = context.prepare_residual_layout(
            resolve_load_regimes(np.zeros(context.experiment_data.strain.shape[0])),
            specs,
        )
        self._weights = weights.copy()
        self._prepared_at_solve = context.solve_iteration
        valid_gate = np.isfinite(weights)
        positive_gate = valid_gate & (weights > 0.0)
        transition_gate = positive_gate & (weights < 1.0)
        positive = weights[positive_gate]
        finite_weights = np.where(valid_gate, weights, 0.0)
        weight_sum = float(np.sum(finite_weights))
        effective_count = weight_sum**2 / max(
            float(np.sum(finite_weights**2)), np.finfo(float).eps
        )

        def captured_fraction(values: FloatArray) -> float:
            resolved = np.where(np.isfinite(values), values, 0.0)
            return float(
                np.sum(finite_weights * resolved)
                / max(float(np.sum(resolved)), np.finfo(float).eps)
            )
        self._diagnostics = {
            "mode": "simple_two_perturbation_gate",
            "phase_index": context.phase_index,
            "prepared_at_solve": context.solve_iteration,
            "refreshed": True,
            "parameter_names": list(self.config.parameter_names),
            "stress_reconstructions": len(self.config.parameter_names),
            "perturbation_factor": self.config.perturbation_factor,
            "gate_start": gate_start,
            "gate_full": gate_full,
            "gate_start_quantile": self.config.gate_start_quantile,
            "gate_full_quantile": self.config.gate_full_quantile,
            "positive_activity_floor": self.config.positive_activity_floor,
            "gate_positive_fraction": float(np.sum(positive_gate) / np.sum(valid_gate)),
            "gate_positive_fraction_of_full_grid": float(np.mean(positive_gate)),
            "gate_transition_fraction_of_positive": float(
                np.sum(transition_gate) / max(np.sum(positive_gate), 1)
            ),
            "gate_effective_sample_fraction": float(effective_count / np.sum(valid_gate)),
            "gate_mean_positive": float(np.mean(positive)),
            "parameter_activity_capture": {
                name: captured_fraction(values) for name, values in scaled.items()
            },
            "objective_weights": {
                "informative_egi": 1.0 - self.config.force_weight - self.config.broad_guard_weight,
                "fre_guard": self.config.force_weight,
                "broad_egi_guard": self.config.broad_guard_weight,
            },
            "residual_layout": self._layout.diagnostics(),
        }
        if self.diagnostic_callback is not None:
            self.diagnostic_callback("simple_sensitivity_gate", {
                **self._diagnostics,
                "weights": weights.copy(),
                "parameter_activity": {name: values.copy() for name, values in activity.items()},
                "parameter_activity_scaled": {name: values.copy() for name, values in scaled.items()},
            })
        return dict(self._diagnostics)

    def _resolve_gate_thresholds(self, combined: FloatArray) -> tuple[float, float]:
        if self.config.gate_start_quantile is None:
            return self.config.gate_start, self.config.gate_full
        positive = combined[np.isfinite(combined) & (combined > 0.0)]
        if positive.size < 2:
            raise ValueError("Sensitivity activity has too few positive observations for quantile gating.")
        start, full = np.quantile(
            positive,
            (self.config.gate_start_quantile, self.config.gate_full_quantile),
        )
        if not np.isfinite(start) or not np.isfinite(full) or full <= start:
            raise ValueError("Sensitivity activity quantiles do not define a non-zero gate ramp.")
        return float(start), float(full)

    def evaluate(self, metric_results) -> float:
        if self._layout is None:
            raise RuntimeError("SensitivityGatedEgiObjective must be prepared before evaluation.")
        vector = self._layout.evaluate(metric_results)
        slices = {name: slice(start, stop) for name, start, stop in vector.block_slices}
        costs = {
            role: float(np.linalg.norm(vector.weighted[slices[f"egi_{role}_informative"]]))
            for role in ("fine", "middle", "broad")
        }
        informative = float(np.mean(list(costs.values())))
        force = float(np.linalg.norm(vector.weighted[slices["fre_guard"]]))
        broad_guard = float(np.linalg.norm(vector.weighted[slices["egi_broad_guard"]]))
        training_weight = 1.0 - self.config.force_weight - self.config.broad_guard_weight
        total = (
            training_weight * informative
            + self.config.force_weight * force
            + self.config.broad_guard_weight * broad_guard
        )
        self.last_result = SensitivityGatedObjectiveResult(
            total, informative, costs["fine"], costs["middle"], costs["broad"],
            force, broad_guard,
        )
        return float(total)

    def diagnostics(self) -> dict[str, object]:
        result = dict(self._diagnostics)
        if self.last_result is not None:
            training_weight = 1.0 - self.config.force_weight - self.config.broad_guard_weight
            result["last_costs"] = {
                "total": self.last_result.total_cost,
                "informative_egi": self.last_result.informative_egi_cost,
                "egi_fine": self.last_result.fine_cost,
                "egi_middle": self.last_result.middle_cost,
                "egi_broad": self.last_result.broad_cost,
                "fre_guard": self.last_result.force_guard_cost,
                "broad_egi_guard": self.last_result.broad_guard_cost,
                "weighted_contributions": {
                    "informative_egi": training_weight * self.last_result.informative_egi_cost,
                    "fre_guard": self.config.force_weight * self.last_result.force_guard_cost,
                    "broad_egi_guard": self.config.broad_guard_weight * self.last_result.broad_guard_cost,
                },
            }
        return result


def _normalise_activity(values: npt.ArrayLike, percentile: float) -> FloatArray:
    activity = np.asarray(values, dtype=np.float64)
    result = np.full(activity.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(activity)
    positive = activity[valid & (activity > 0.0)]
    if positive.size == 0:
        result[valid] = 0.0
        return result
    scale = float(np.percentile(positive, percentile))
    result[valid] = np.clip(activity[valid] / scale, 0.0, 1.0)
    return result


def _smooth_gate(activity: FloatArray, start: float, full: float) -> FloatArray:
    position = np.clip((activity - start) / (full - start), 0.0, 1.0)
    return position * position * (3.0 - 2.0 * position)
