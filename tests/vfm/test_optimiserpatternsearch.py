"""Focused tests for the parallel generalized pattern search."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
import pytest

from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.optimiserpatternsearch import OptimiserPatternSearch
from pyvale.vfm.spatialparamhomogeneous import SpatialParameterisationHomogeneous


def _run_search(
    monkeypatch: pytest.MonkeyPatch,
    objective: Callable[[np.ndarray], float],
    initial: np.ndarray,
    **options,
):
    monkeypatch.setattr(
        "pyvale.vfm.optimiserpatternsearch.evaluate_candidate",
        lambda candidate, *_args: objective(np.asarray(candidate)),
    )
    spatial = {
        f"x{index}": [
            SpatialParameterisationHomogeneous(
                DegreeOfFreedom(float(value), 0.0, 1.0),
            )
        ]
        for index, value in enumerate(initial)
    }
    outcome = OptimiserPatternSearch(**options).optimise(
        None,
        np.asarray((1, 1), dtype=np.uint32),
        spatial,
        [],
        type("Objective", (), {"last_result": None})(),
        None,
    )
    final = np.asarray([
        outcome.spatial_parameterisations[f"x{index}"][0].value.value
        for index in range(initial.size)
    ])
    return outcome, final


def test_rotated_poll_bases_reduce_correlated_quadratic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    number_of_variables = 5
    rotation, _ = np.linalg.qr(
        np.random.default_rng(7).normal(
            size=(number_of_variables, number_of_variables),
        )
    )
    curvature = rotation @ np.diag(
        np.geomspace(1.0, 1_000.0, number_of_variables)
    ) @ rotation.T
    target = np.linspace(0.2, 0.8, number_of_variables)

    def objective(candidate: np.ndarray) -> float:
        residual = candidate - target
        return float(residual @ curvature @ residual)

    outcome, _ = _run_search(
        monkeypatch,
        objective,
        np.full(number_of_variables, 0.9),
        initial_mesh_size=0.25,
        minimum_mesh_size=1.0e-5,
        max_iterations=1_000,
        max_evaluations=1_000,
        random_seed=1,
    )

    assert outcome.solve_result is not None
    assert outcome.solve_result.final_objective["cost"] < 1.0e-2
    history = outcome.solve_result.final_objective["history"]
    assert {entry["basis"] for entry in history[:2]} == {
        "coordinate",
        "orthonormal",
    }


def test_relative_tolerance_preserves_small_scale_improvements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = np.asarray((0.2, 0.5, 0.8))

    def objective(candidate: np.ndarray) -> float:
        return 1.0e-14 * float(np.sum((candidate - target) ** 2))

    outcome, final = _run_search(
        monkeypatch,
        objective,
        np.full(3, 0.9),
        initial_mesh_size=0.25,
        minimum_mesh_size=1.0e-5,
        max_iterations=500,
        max_evaluations=500,
    )

    assert outcome.solve_result is not None
    assert outcome.solve_result.final_objective["cost"] < 1.0e-22
    np.testing.assert_allclose(final, target, atol=2.0e-4)


def test_seeded_orthonormal_polls_are_reproducible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = np.asarray((0.15, 0.45, 0.85))
    objective = lambda candidate: float(np.sum((candidate - target) ** 2))
    options = {
        "initial_mesh_size": 0.2,
        "minimum_mesh_size": 1.0e-4,
        "max_iterations": 20,
        "max_evaluations": 150,
        "random_seed": 42,
    }

    first, first_final = _run_search(
        monkeypatch, objective, np.full(3, 0.9), **options,
    )
    second, second_final = _run_search(
        monkeypatch, objective, np.full(3, 0.9), **options,
    )

    np.testing.assert_array_equal(first_final, second_final)
    assert first.solve_result is not None
    assert second.solve_result is not None
    assert (
        first.solve_result.final_objective["history"]
        == second.solve_result.final_objective["history"]
    )


def test_max_batch_size_limits_simultaneous_evaluations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def objective(candidate: np.ndarray) -> float:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return float(np.sum((candidate - 0.4) ** 2))

    outcome, _ = _run_search(
        monkeypatch,
        objective,
        np.full(4, 0.8),
        max_iterations=1,
        max_evaluations=9,
        parallel_workers=5,
        max_batch_size=2,
    )

    assert outcome.solve_result is not None
    assert maximum_active == 2


def test_pattern_candidate_continues_successful_displacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome, final = _run_search(
        monkeypatch,
        lambda candidate: float((candidate[0] - 0.8) ** 2),
        np.asarray((0.2,)),
        initial_mesh_size=0.1,
        minimum_mesh_size=1.0e-4,
        max_iterations=3,
        max_evaluations=20,
        pattern_step_size=2.0,
    )

    assert outcome.solve_result is not None
    history = outcome.solve_result.final_objective["history"]
    assert any(
        entry["accepted_candidate"] == "pattern" for entry in history
    )
    assert final[0] > 0.4


@pytest.mark.parametrize(
    ("mesh_expansion_factor", "expected_mesh_size"),
    [(1.0, 0.1), (2.0, 0.2)],
)
def test_mesh_expansion_is_explicit_and_applied_after_improvement(
    monkeypatch: pytest.MonkeyPatch,
    mesh_expansion_factor: float,
    expected_mesh_size: float,
) -> None:
    outcome, _ = _run_search(
        monkeypatch,
        lambda candidate: float((candidate[0] - 0.8) ** 2),
        np.asarray((0.2,)),
        initial_mesh_size=0.1,
        minimum_mesh_size=1.0e-4,
        max_iterations=1,
        max_evaluations=3,
        mesh_expansion_factor=mesh_expansion_factor,
    )

    assert outcome.solve_result is not None
    history = outcome.solve_result.final_objective["history"]
    assert history[0]["mesh_size"] == pytest.approx(expected_mesh_size)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("pattern_step_size", -1.0),
        ("objective_absolute_tolerance", -1.0),
        ("objective_relative_tolerance", -1.0),
        ("mesh_contraction_factor", 1.0),
        ("mesh_expansion_factor", 0.5),
        ("max_batch_size", 0),
    ],
)
def test_invalid_generalized_search_options_are_rejected(
    option: str,
    value: float,
) -> None:
    with pytest.raises(ValueError):
        OptimiserPatternSearch(**{option: value})
