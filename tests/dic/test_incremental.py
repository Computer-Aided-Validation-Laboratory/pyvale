"""Tests for incremental DIC reference-update conditions."""

import pytest

import pyvale.dic.diccpp as diccpp


@pytest.mark.parametrize(
    ("condition", "image_number", "iterations", "costs", "threshold", "expected"),
    [
        # The first deformed image never updates the reference.
        ("IMAGE", 1, [4], [0.5], 1, False),
        ("IMAGE", 2, [4], [0.5], 1, True),
        ("IMAGE", 3, [4], [0.5], 2, True),
        # COST updates only when the mean ZNCC cost is strictly below the threshold.
        ("COST", 2, [4, 4], [0.4, 0.6], 0.6, True),
        ("COST", 2, [4, 4], [0.4, 0.6], 0.5, False),
        ("COST", 2, [4, 4], [0.4, 0.6], 0.4, False),
        # ITER updates only when the mean iteration count exceeds the threshold.
        ("ITER", 2, [4, 6], [0.5, 0.5], 4.9, True),
        ("ITER", 2, [4, 6], [0.5, 0.5], 5, False),
    ],
)
def test_incremental_update_conditions(
    condition: str,
    image_number: int,
    iterations: list[int],
    costs: list[float],
    threshold: float,
    expected: bool,
) -> None:
    result = diccpp._should_update_ref(
        image_number,
        iterations,
        costs,
        getattr(diccpp.IncrementalCond, condition),
        threshold,
    )

    assert result is expected
