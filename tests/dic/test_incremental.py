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


def test_incremental_cost_is_independent_of_output_filtering(
    tmp_path: Path,
) -> None:
    """Output masking must not alter the state used for COST updates."""
    filtered_dir = tmp_path / "filtered"
    unfiltered_dir = tmp_path / "unfiltered"
    filtered_dir.mkdir()
    unfiltered_dir.mkdir()

    common_args = {
        "reference": ref0,
        "deformed": def0,
        "roi_mask": roi.mask,
        "seed": seed,
        "subset_size": 31,
        "subset_step": 15,
        "max_displacement": 60,
        "correlation_criteria": "ZNSSD",
        "shape_function": "AFFINE",
        "method": "MULTIWINDOW_RG",
        "incremental_update": "COST",
        "incremental_update_value": 1.0,
        "threshold": 0.99,
        "image_filter_kernel": 1,
    }

    dic.calculate_2d(
        **common_args,
        output_basepath=filtered_dir,
        output_prefix="filtered_",
        output_below_threshold=False,
    )
    dic.calculate_2d(
        **common_args,
        output_basepath=unfiltered_dir,
        output_prefix="unfiltered_",
        output_below_threshold=True,
    )

    filtered_files = sorted(filtered_dir.glob("filtered_*.csv"))
    unfiltered_files = sorted(unfiltered_dir.glob("unfiltered_*.csv"))
    assert len(filtered_files) == len(unfiltered_files) > 1

    found_filtered_point = False
    for filtered_file, unfiltered_file in zip(filtered_files, unfiltered_files):
        filtered = np.loadtxt(filtered_file, skiprows=1, delimiter=",")
        unfiltered = np.loadtxt(unfiltered_file, skiprows=1, delimiter=",")
        valid = np.isfinite(filtered[:, 2])
        found_filtered_point |= np.any(~valid)
        np.testing.assert_allclose(
            filtered[valid, 2:4],
            unfiltered[valid, 2:4],
            atol=1e-6,
            rtol=0.0,
        )

    assert found_filtered_point
