import numpy as np
import pytest

from pyvale.vfm.metric import MetricResult
from pyvale.vfm.objectivefunc import IScalarObjectiveFunction
from pyvale.vfm.objectivefuncmaterialinformation import (
    MaterialFeatureReduction,
    MaterialFeatureReference,
    MaterialFeatureTerm,
    MaterialInformationObjective,
)
from pyvale.vfm.objectivefunccombinedfreegi import (
    CombinedForceAndEquilibriumGapObjective,
)
from pyvale.vfm.refinement import _global_combined_objective


class ConstantObjective(IScalarObjectiveFunction):
    def __init__(self, value):
        self.value = value

    def evaluate(self, metric_results):
        return self.value


def _metric(values):
    return MetricResult(
        residual=np.asarray(values, dtype=float),
        additional_fields={"normalised_gap": np.asarray(values, dtype=float)},
    )


def _objective(alpha=0.5):
    return MaterialInformationObjective(
        global_objective=ConstantObjective(2.0),
        feature_terms=[
            MaterialFeatureTerm("tail", 0, MaterialFeatureReduction.CVAR_ABS, quantile=0.5)
        ],
        alpha=alpha,
        mean_fraction=0.0,
        smooth_max_temperature=0.1,
        positive_part_temperature=1e-6,
        references={"tail": MaterialFeatureReference(1.0, 5.0)},
    )


def test_alpha_zero_reproduces_global_objective_without_references():
    objective = MaterialInformationObjective(
        global_objective=ConstantObjective(2.5), feature_terms=[], alpha=0.0
    )
    assert objective.evaluate([]) == 2.5


def test_stage_reference_normalises_feature_to_one():
    objective = _objective()
    value = objective.evaluate([_metric([5.0, 5.0])])
    assert objective.last_result.features[0].normalised_value == pytest.approx(1.0)
    assert value == pytest.approx(1.5)


def test_noise_floor_normalises_feature_near_zero():
    objective = _objective(alpha=1.0)
    assert objective.evaluate([_metric([1.0, 1.0])]) < 1e-5


def test_worsening_dominant_feature_increases_objective_and_diagnostics_reconstruct():
    objective = _objective(alpha=0.75)
    before = objective.evaluate([_metric([3.0, 3.0])])
    after = objective.evaluate([_metric([7.0, 7.0])])
    assert after > before
    result = objective.last_result
    assert result.total_cost == pytest.approx(
        (1.0 - objective.alpha) * result.global_cost
        + objective.alpha * result.material_cost
    )


def test_capture_stage_references_uses_configured_noise_floor():
    objective = _objective()
    captured = objective.capture_stage_references(
        [_metric([4.0, 6.0])], noise_floors={"tail": 0.5}
    )
    assert captured["tail"].noise_floor == 0.5
    assert captured["tail"].stage_reference == pytest.approx(6.0)


def test_hybrid_exposes_global_closure_to_sensitivity_basis_growth():
    closure = CombinedForceAndEquilibriumGapObjective(
        egi_window_weights=(1.0, 1.0)
    )
    hybrid = MaterialInformationObjective(
        global_objective=closure, feature_terms=[], alpha=0.0
    )
    assert _global_combined_objective(hybrid) is closure
