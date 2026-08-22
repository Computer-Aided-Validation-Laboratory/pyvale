import numpy as np

from pyvale.vfm.metric import MetricResult
from pyvale.vfm.objectivefunccombinedfreegi import (
    CombinedForceAndEquilibriumGapObjective,
    infer_egi_window_length_weights,
)


def _force_result(value: float) -> MetricResult:
    return MetricResult(
        residual=np.array([value]),
        additional_fields={
            "normalised_residual": np.array([[value]]),
            "temporal_weights": np.array([1.0]),
            "spatial_weights": np.array([1.0]),
            "reconstructed_force": np.array([[1.0]]),
        },
    )


def _egi_result(value: float, window: int) -> MetricResult:
    return MetricResult(
        residual=np.array([value]),
        additional_fields={
            "weighted_spatiotemporal_rms": value,
            "window_size": np.array([window, window]),
        },
    )


def test_combined_objective_matches_reference_equations() -> None:
    objective = CombinedForceAndEquilibriumGapObjective(
        force_weight=0.1,
        egi_baseline_values=(2.0, 4.0),
        force_baseline_value=5.0,
        egi_window_weights=(1.0, 2.0),
    )

    value = objective.evaluate([
        _force_result(10.0),
        _egi_result(4.0, 29),
        _egi_result(8.0, 57),
    ])

    # Both EGI ratios are 2 and FRE ratio is 2, so phi must be 2.
    assert value == 2.0
    assert objective.last_result is not None
    assert objective.last_result.equilibrium_gap_cost == 2.0
    assert objective.last_result.force_cost == 2.0


def test_egi_window_weights_use_length_not_area() -> None:
    weights = infer_egi_window_length_weights([
        _egi_result(1.0, 29),
        _egi_result(1.0, 57),
    ])
    np.testing.assert_allclose(weights, np.array([29.0, 57.0]) / 86.0)


def test_combined_objective_is_one_at_baseline_and_force_weight_limits() -> None:
    results = [_force_result(5.0), _egi_result(2.0, 29), _egi_result(4.0, 57)]
    common = dict(
        egi_baseline_values=(2.0, 4.0),
        force_baseline_value=5.0,
        egi_window_weights=(29.0, 57.0),
    )
    assert CombinedForceAndEquilibriumGapObjective(force_weight=0.1, **common).evaluate(results) == 1.0

    # Candidate EGI is twice baseline while FRE is three times baseline.
    candidate = [_force_result(15.0), _egi_result(4.0, 29), _egi_result(8.0, 57)]
    assert CombinedForceAndEquilibriumGapObjective(force_weight=0.0, **common).evaluate(candidate) == 2.0
    assert CombinedForceAndEquilibriumGapObjective(force_weight=1.0, **common).evaluate(candidate) == 3.0


def test_combined_objective_accepts_zero_candidate_egi() -> None:
    objective = CombinedForceAndEquilibriumGapObjective(
        force_weight=0.0,
        egi_baseline_values=(2.0,),
        force_baseline_value=5.0,
        egi_window_weights=(1.0,),
    )

    assert objective.evaluate([_force_result(5.0), _egi_result(0.0, 29)]) == 0.0


def test_window_weights_change_only_egi_contribution() -> None:
    results = [_force_result(15.0), _egi_result(4.0, 29), _egi_result(4.0, 57)]
    common = dict(
        force_weight=0.25,
        egi_baseline_values=(2.0, 4.0),
        force_baseline_value=5.0,
    )
    first_window = CombinedForceAndEquilibriumGapObjective(
        egi_window_weights=(3.0, 1.0), **common
    )
    second_window = CombinedForceAndEquilibriumGapObjective(
        egi_window_weights=(1.0, 3.0), **common
    )

    first_window.evaluate(results)
    second_window.evaluate(results)

    assert first_window.last_result is not None
    assert second_window.last_result is not None
    assert first_window.last_result.force_cost == second_window.last_result.force_cost
    assert first_window.last_result.equilibrium_gap_cost == 1.75
    assert second_window.last_result.equilibrium_gap_cost == 1.25


def test_scalar_egi_aggregation_is_not_rms_of_a_combined_map() -> None:
    egi_map_a = np.array([0.0, 2.0])
    egi_map_b = np.array([2.0, 0.0])
    scalar_egi = float(np.sqrt(np.mean(egi_map_a**2)))
    objective = CombinedForceAndEquilibriumGapObjective(
        force_weight=0.0,
        egi_baseline_values=(1.0, 1.0),
        force_baseline_value=1.0,
        egi_window_weights=(1.0, 1.0),
    )

    scalar_cost = objective.evaluate([
        _force_result(1.0),
        _egi_result(scalar_egi, 29),
        _egi_result(scalar_egi, 57),
    ])
    combined_map_rms = float(np.sqrt(np.mean(((egi_map_a + egi_map_b) / 2.0) ** 2)))

    assert scalar_cost == scalar_egi
    assert combined_map_rms == 1.0
    assert scalar_cost != combined_map_rms
