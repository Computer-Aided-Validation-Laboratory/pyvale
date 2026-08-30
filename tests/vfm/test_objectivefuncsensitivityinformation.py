from types import SimpleNamespace

import numpy as np
import pytest

from pyvale.vfm.loadregimes import resolve_load_regimes
from pyvale.vfm.metric import MetricResult
from pyvale.vfm.objectivefuncsensitivityinformation import (
    SensitivityInformationObjective,
    SensitivityInformationObjectiveConfig,
)
from pyvale.vfm.residualblocks import (
    ResidualBlockSpec,
    prepare_canonical_residual_layout,
)


class _LinearContext:
    def __init__(self) -> None:
        self.phase_index = 1
        self.solve_iteration = 0
        self.normalised_degrees_of_freedom = np.array([0.5])
        self.metric_results = tuple(self.evaluate_metric_results(self.normalised_degrees_of_freedom))

    def evaluate_metric_results(self, dofs):
        value = float(np.asarray(dofs)[0] - 0.5)
        return [
            MetricResult(residual=np.full((2, 2), value)),
            MetricResult(residual=np.full((2, 2), 2.0 * value)),
        ]

    def prepare_residual_layout(self, load_regimes, specs):
        return prepare_canonical_residual_layout(
            self.metric_results, load_regimes, specs
        )


def _config(**overrides) -> SensitivityInformationObjectiveConfig:
    values = {
        "load_regimes": resolve_load_regimes((0.0, 1.0)),
        "residual_blocks": (
            ResidualBlockSpec("fre", 0, "all", "fre", role="fre_guard"),
            ResidualBlockSpec("broad", 1, "all", "egi", role="broad_egi_guard"),
        ),
        "meaningful_dof_movement": 1.0,
        "minimum_noise_response": 0.1,
        "robust_transition": 1.0,
    }
    values.update(overrides)
    return SensitivityInformationObjectiveConfig(**values)


def test_prepared_objective_requires_preparation_and_keeps_equal_guards() -> None:
    objective = SensitivityInformationObjective(_config())
    context = _LinearContext()

    with pytest.raises(RuntimeError, match="prepared"):
        objective.evaluate(context.evaluate_metric_results([0.6]))

    diagnostics = objective.prepare_solve(context)
    cost = objective.evaluate(context.evaluate_metric_results([0.6]))

    assert diagnostics["retained_rank"] == 1
    assert objective.last_result is not None
    assert cost == pytest.approx(objective.last_result.total_cost)
    assert objective.last_result.fre_guard_cost > 0.0
    assert objective.last_result.broad_egi_guard_cost > 0.0
    assert objective.preparation_count == 1


def test_absolute_rank_gate_can_leave_only_mechanical_guards() -> None:
    objective = SensitivityInformationObjective(
        _config(minimum_noise_response=100.0)
    )
    context = _LinearContext()

    objective.prepare_solve(context)
    objective.evaluate(context.evaluate_metric_results([0.6]))

    assert objective.last_result is not None
    assert objective.last_result.retained_rank == 0
    assert objective.last_result.material_cost == 0.0
    assert objective.last_result.total_cost > 0.0


def test_prepare_solve_is_repeatable_at_the_same_state() -> None:
    objective = SensitivityInformationObjective(_config())
    context = _LinearContext()

    first = objective.prepare_solve(context)
    second = objective.prepare_solve(context)

    assert first["singular_values"] == pytest.approx(second["singular_values"])
    assert first["retained_rank"] == second["retained_rank"]
    assert objective.preparation_count == 2
