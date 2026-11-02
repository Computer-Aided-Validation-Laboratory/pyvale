"""Tests for incremental DIC reference-update conditions."""

from pathlib import Path

import numpy as np
import pytest

import pyvale.dic as dic
import pyvale.dic.diccpp as diccpp
import pyvale.dic._dicchecks as dicchecks
import pyvale.data as dataset


ref0 = dataset.dic_plate_rigid_cam0_ref()
def0 = dataset.dic_plate_rigid_cam0_def_small()

roi = dic.RegionOfInterest(ref0)
roi.rect_boundary(left=0, right=100, bottom=100, top=0)
seed = [50, 200]


@pytest.mark.parametrize(
    ("condition", "value"),
    [
        ("IMAGE", 0),
        ("IMAGE", -1),
        ("IMAGE", 1.5),
        ("COST", float("nan")),
        ("ITER", float("inf")),
    ],
)
def test_invalid_incremental_update_value(condition: str, value: float | int) -> None:
    with pytest.raises(ValueError, match="incremental_update_value"):
        dicchecks._check_incremental_update_value(condition, value)


@pytest.mark.parametrize(
    ("condition", "value"),
    [("IMAGE", 1), ("IMAGE", 2.0), ("COST", 0.9), ("ITER", 10)],
)
def test_valid_incremental_update_value(condition: str, value: float | int) -> None:
    dicchecks._check_incremental_update_value(condition, value)


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
