from types import SimpleNamespace

import numpy as np
import pytest

from pyvale.vfm.metric import MetricResult
from pyvale.vfm.objectivefuncsensitivitygated import (
    SensitivityGatedEgiObjective,
    SensitivityGatedObjectiveConfig,
)
from pyvale.vfm.residualblocks import prepare_canonical_residual_layout


class _Context:
    def __init__(self) -> None:
        self.phase_index = 1
        self.solve_iteration = 0
        self.experiment_data = SimpleNamespace(strain=np.zeros((2, 3, 2, 2)))
        self.stress = np.zeros((2, 3, 2, 2))
        self.constitutive_law = object()
        self.parameter_maps = {
            "yield_strength": np.ones((2, 2)),
            "hardening_modulus": np.ones((2, 2)),
        }
        self.metric_results = tuple(_metric_results(1.0))

    def prepare_residual_layout(self, load_regimes, specs):
        return prepare_canonical_residual_layout(
            self.metric_results, load_regimes, specs
        )


def _metric_results(scale: float):
    force = np.full((2, 2), scale)
    gap = np.full((2, 2, 2), scale)
    return [
        MetricResult(
            residual=force,
            additional_fields={"normalised_residual": force},
        ),
        *[
            MetricResult(
                residual=gap,
                additional_fields={"normalised_gap": gap},
            )
            for _ in range(3)
        ],
    ]


def test_simple_objective_uses_two_perturbations_and_explicit_guards(monkeypatch) -> None:
    calls = []

    def fake_sensitivities(strain, stress, law, maps, names, perturbation_factor):
        calls.extend(names)
        first = np.zeros_like(stress)
        first[:, 0, :, 0] = 1.0
        second = np.zeros_like(stress)
        second[1, 1, :, 1] = 2.0
        return {
            names[0]: SimpleNamespace(total=first),
            names[1]: SimpleNamespace(total=second),
        }

    monkeypatch.setattr(
        "pyvale.vfm.objectivefuncsensitivitygated.calculate_parameter_stress_sensitivities",
        fake_sensitivities,
    )
    objective = SensitivityGatedEgiObjective(
        SensitivityGatedObjectiveConfig(
            gate_start=0.0,
            gate_full=0.5,
            egi_noise_scales=(2.0, 2.0, 2.0),
            force_noise_scale=4.0,
            force_weight=0.2,
            broad_guard_weight=0.1,
        )
    )
    context = _Context()

    diagnostics = objective.prepare_solve(context)
    cost = objective.evaluate(_metric_results(1.0))

    assert calls == ["yield_strength", "hardening_modulus"]
    assert diagnostics["mode"] == "simple_two_perturbation_gate"
    assert diagnostics["stress_reconstructions"] == 2
    assert objective.last_result is not None
    assert objective.last_result.informative_egi_cost == pytest.approx(0.5)
    assert objective.last_result.force_guard_cost == pytest.approx(0.25)
    assert objective.last_result.broad_guard_cost == pytest.approx(0.5)
    assert cost == pytest.approx(0.7 * 0.5 + 0.2 * 0.25 + 0.1 * 0.5)


def test_simple_objective_freezes_gate_between_solves(monkeypatch) -> None:
    call_count = 0

    def fake_sensitivities(strain, stress, law, maps, names, perturbation_factor):
        nonlocal call_count
        call_count += 1
        values = np.ones_like(stress)
        return {name: SimpleNamespace(total=values) for name in names}

    monkeypatch.setattr(
        "pyvale.vfm.objectivefuncsensitivitygated.calculate_parameter_stress_sensitivities",
        fake_sensitivities,
    )
    objective = SensitivityGatedEgiObjective(SensitivityGatedObjectiveConfig())
    context = _Context()

    objective.prepare_solve(context)
    context.solve_iteration = 1
    diagnostics = objective.prepare_solve(context)

    assert call_count == 1
    assert diagnostics["refreshed"] is False


def test_quantile_gate_resolves_from_positive_activity(monkeypatch) -> None:
    def fake_sensitivities(strain, stress, law, maps, names, perturbation_factor):
        values = np.zeros_like(stress)
        values[:, 0] = np.array(((0.0, 1.0), (2.0, 4.0)))
        return {name: SimpleNamespace(total=values) for name in names}

    monkeypatch.setattr(
        "pyvale.vfm.objectivefuncsensitivitygated.calculate_parameter_stress_sensitivities",
        fake_sensitivities,
    )
    objective = SensitivityGatedEgiObjective(
        SensitivityGatedObjectiveConfig(
            gate_start_quantile=0.0,
            gate_full_quantile=0.5,
        )
    )

    diagnostics = objective.prepare_solve(_Context())

    assert diagnostics["gate_start"] > 0.0
    assert diagnostics["gate_full"] > diagnostics["gate_start"]
    assert diagnostics["gate_start_quantile"] == 0.0
    assert diagnostics["gate_full_quantile"] == 0.5
