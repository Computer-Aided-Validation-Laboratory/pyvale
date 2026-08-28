import numpy as np
import pytest

from pyvale.vfm.equilibriumgapaggregation import aggregate_equilibrium_gap_results
from pyvale.vfm.metric import MetricResult
from pyvale.vfm.objectivefunccombinedfreegi import (
    CombinedForceAndEquilibriumGapObjective,
    CombinedObjectiveBaseline,
    infer_egi_window_length_weights,
)
from pyvale.vfm.spatialweighting import (
    SensitivitySpatialWeightingConfig,
    SensitivitySpatialWeights,
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


def _detailed_egi_result(values: np.ndarray, window: int) -> MetricResult:
    resolved = np.asarray(values, dtype=np.float64)
    return MetricResult(
        residual=resolved,
        additional_fields={
            "normalised_gap": resolved,
            "temporal_weights": np.ones(resolved.shape[0]),
            "weighted_temporal_rms": np.sqrt(np.mean(resolved**2, axis=0)),
            "weighted_spatiotemporal_rms": float(np.sqrt(np.mean(resolved**2))),
            "window_size": np.array([window, window]),
        },
    )


def _detailed_force_result(values: np.ndarray) -> MetricResult:
    resolved = np.asarray(values, dtype=np.float64)
    temporal = np.array([0.25, 0.75])
    spatial = np.array([0.4, 0.6])
    return MetricResult(
        residual=resolved,
        additional_fields={
            "normalised_residual": resolved,
            "temporal_weights": temporal,
            "spatial_weights": spatial,
            "reconstructed_force": np.ones_like(resolved),
        },
    )


def _set_resolved_weights(
    objective: CombinedForceAndEquilibriumGapObjective,
    *,
    egi_weights: tuple[np.ndarray, ...],
    force_weights: np.ndarray,
) -> None:
    objective.resolved_spatial_weights = SensitivitySpatialWeights(
        parameter_names=("yield_strength",),
        equilibrium_gap_weights=egi_weights,
        force_weights=force_weights,
        equilibrium_gap_parameter_activity=tuple(
            {"yield_strength": np.ones_like(weights)}
            for weights in egi_weights
        ),
        force_parameter_activity={
            "yield_strength": np.ones_like(force_weights),
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


def test_combined_objective_residual_cotangents_match_finite_difference() -> None:
    objective = CombinedForceAndEquilibriumGapObjective(
        force_weight=0.2,
        egi_baseline_values=(0.7,),
        force_baseline_value=0.5,
        egi_window_weights=(1.0,),
    )
    egi_values = np.array([[[0.2, -0.1]], [[0.4, 0.3]]])
    force_values = np.array([[0.1, -0.2], [0.3, 0.25]])
    egi_direction = np.array([[[0.3, 0.1]], [[-0.2, 0.4]]])
    force_direction = np.array([[0.2, -0.1], [0.4, 0.3]])
    results = [_detailed_force_result(force_values), _detailed_egi_result(egi_values, 29)]
    cotangents = objective.residual_cotangents(results)
    predicted = float(
        np.sum(cotangents.equilibrium_gap[0] * egi_direction)
        + np.sum(cotangents.force * force_direction)
    )
    step = 1.0e-6
    plus = objective.evaluate([
        _detailed_force_result(force_values + step * force_direction),
        _detailed_egi_result(egi_values + step * egi_direction, 29),
    ])
    minus = objective.evaluate([
        _detailed_force_result(force_values - step * force_direction),
        _detailed_egi_result(egi_values - step * egi_direction, 29),
    ])

    assert predicted == pytest.approx((plus - minus) / (2.0 * step), rel=1.0e-6)


def test_combined_objective_uses_unit_baselines_by_default() -> None:
    objective = CombinedForceAndEquilibriumGapObjective(
        force_weight=0.25,
        egi_window_weights=(1.0, 1.0),
    )

    value = objective.evaluate([
        _force_result(8.0),
        _egi_result(2.0, 29),
        _egi_result(6.0, 57),
    ])

    assert value == 5.0
    assert objective.last_result is not None
    np.testing.assert_allclose(objective.last_result.egi_baselines, (1.0, 1.0))
    assert objective.last_result.force_baseline == 1.0
    np.testing.assert_allclose(objective.egi_baselines_for(2), (1.0, 1.0))


def test_combined_objective_resolves_prior_phase_baselines() -> None:
    objective = CombinedForceAndEquilibriumGapObjective(
        force_weight=0.1,
        egi_window_weights=(29.0, 57.0),
        baseline=CombinedObjectiveBaseline.prior_phase(0),
    )
    reference = [_force_result(5.0), _egi_result(2.0, 29), _egi_result(4.0, 57)]
    objective.resolve_from_prior_phase(reference)

    assert objective.evaluate(reference) == 1.0
    assert objective.baseline_diagnostics() == {
        "mode": "prior_phase",
        "phase_index": 0,
        "egi_values": [2.0, 4.0],
        "force_value": 5.0,
    }
    np.testing.assert_allclose(objective.egi_baselines_for(2), (2.0, 4.0))


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


def test_combined_objective_accepts_compact_metric_results() -> None:
    objective = CombinedForceAndEquilibriumGapObjective(
        force_weight=0.25,
        egi_baseline_values=(2.0, 4.0),
        force_baseline_value=5.0,
        egi_window_weights=(1.0, 3.0),
    )
    compact_results = [
        MetricResult(
            additional_fields={
                "reconstructed_force": np.zeros(1),
                "normalised_residual": np.asarray([15.0]),
            }
        ),
        MetricResult(
            additional_fields={
                "weighted_spatiotemporal_rms": 4.0,
                "window_size": np.asarray([5, 5]),
            }
        ),
        MetricResult(
            additional_fields={
                "weighted_spatiotemporal_rms": 8.0,
                "window_size": np.asarray([9, 9]),
            }
        ),
    ]

    assert objective.evaluate(compact_results) == 2.25


def test_sensitivity_weights_apply_sqrt_weight_to_egi_and_fre_residuals() -> None:
    objective = CombinedForceAndEquilibriumGapObjective(
        force_weight=0.5,
        egi_window_weights=(1.0,),
        spatial_weighting=SensitivitySpatialWeightingConfig(),
    )
    _set_resolved_weights(
        objective,
        egi_weights=(np.array([[1.8, 0.2]]),),
        force_weights=np.array([0.9, 0.1]),
    )
    force_result = MetricResult(
        residual=np.array([[1.0, 3.0]]),
        additional_fields={
            "normalised_residual": np.array([[1.0, 3.0]]),
            "temporal_weights": np.array([1.0]),
            "spatial_weights": np.array([0.5, 0.5]),
            "reconstructed_force": np.ones((1, 2)),
        },
    )
    egi_result = _detailed_egi_result(np.array([[[1.0, 3.0]]]), 29)

    value = objective.evaluate([force_result, egi_result])

    expected = np.sqrt(1.8)
    assert value == pytest.approx(expected)
    assert objective.last_result is not None
    assert objective.last_result.force_scalar == pytest.approx(expected)
    np.testing.assert_allclose(objective.last_result.egi_scalars, expected)


def test_weighted_prior_phase_is_one_when_re_evaluated() -> None:
    objective = CombinedForceAndEquilibriumGapObjective(
        force_weight=0.25,
        egi_window_weights=(1.0,),
        baseline=CombinedObjectiveBaseline.prior_phase(0),
        spatial_weighting=SensitivitySpatialWeightingConfig(),
    )
    _set_resolved_weights(
        objective,
        egi_weights=(np.array([[1.5, 0.5]]),),
        force_weights=np.array([0.75, 0.25]),
    )
    results = [
        MetricResult(
            residual=np.array([[2.0, 4.0]]),
            additional_fields={
                "normalised_residual": np.array([[2.0, 4.0]]),
                "temporal_weights": np.array([1.0]),
                "spatial_weights": np.array([0.5, 0.5]),
                "reconstructed_force": np.ones((1, 2)),
            },
        ),
        _detailed_egi_result(np.array([[[2.0, 4.0]]]), 29),
    ]

    objective.resolve_from_prior_phase(results)

    assert objective.evaluate(results) == pytest.approx(1.0)


def test_configured_spatial_weighting_must_be_resolved_before_evaluation() -> None:
    objective = CombinedForceAndEquilibriumGapObjective(
        force_weight=0.1,
        egi_window_weights=(1.0,),
        spatial_weighting=SensitivitySpatialWeightingConfig(),
    )

    with pytest.raises(ValueError, match="have not been resolved"):
        objective.evaluate([_force_result(1.0), _egi_result(1.0, 29)])


def test_egi_aggregation_uses_weighted_local_contribution_for_peak_map() -> None:
    result = _detailed_egi_result(np.array([[[10.0, 9.0]]]), 29)

    aggregation = aggregate_equilibrium_gap_results(
        [result],
        egi_baseline_values=(1.0,),
        window_weights=(1.0,),
        spatial_weights=(np.array([[0.1, 1.9]]),),
    )

    assert aggregation.combined_baseline_scaled_egi_map[0, 1] > (
        aggregation.combined_baseline_scaled_egi_map[0, 0]
    )
    np.testing.assert_allclose(
        aggregation.combined_baseline_scaled_egi_map,
        np.array([[10.0 * np.sqrt(0.1), 9.0 * np.sqrt(1.9)]]),
    )
