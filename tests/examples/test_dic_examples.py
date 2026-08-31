# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Smoke tests for automated documented DIC examples."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

_DIC_EXAMPLES = (
    pytest.param("dic/ex05_dic_challenge.py", "pyvale-output"),
    pytest.param(
        "dic/ex07_incremental.py",
        "pyvale-output/incremental",
        marks=pytest.mark.example_slow,
    ),
)


@pytest.mark.example
@pytest.mark.example_module("dic")
def test_plate_with_hole_examples(run_example: Callable[..., Path]) -> None:
    """The strain example runs against DIC output from the preceding example."""
    work_dir = run_example(
        "dic/ex02_plate_with_hole.py",
        ("pyvale-output/ex02",),
        support_files=("ex10_roi.yaml",),
        timeout=300.0,
    )

    run_example(
        "dic/ex03_plate_with_hole_strain.py",
        ("pyvale-output/ex03",),
        timeout=300.0,
    )

    assert (work_dir / "pyvale-output/ex03").is_dir()


@pytest.mark.example
@pytest.mark.parametrize(("example", "output"), _DIC_EXAMPLES)
def test_dic_example(
    run_example: Callable[..., Path],
    example: str,
    output: str,
) -> None:
    """Each supported standalone DIC gallery example runs successfully."""
    run_example(example, (output,), timeout=300.0)


@pytest.mark.example
@pytest.mark.example_module("dic")
def test_render_to_dic_example(run_example: Callable[..., Path]) -> None:
    """The Riley render produces an accurate analytic DIC displacement."""
    work_dir = run_example(
        "dic/ex04_render_to_dic.py",
        (
            "pyvale-output/dic_ex04_render_to_dic/"
            "dic_ex04_render_to_dic.png",
        ),
        timeout=300.0,
    )

    dic_dir = (
        work_dir
        / "pyvale-output"
        / "dic_ex04_render_to_dic"
        / "dic"
    )
    result_files = sorted(dic_dir.glob("render_to_dic_*.csv"))
    assert len(result_files) == 1

    result = np.loadtxt(result_files[0], delimiter=",", skiprows=1)
    assert np.all(result[:, 5].astype(bool))
    np.testing.assert_allclose(result[:, 2], 0.5, atol=0.05)
    np.testing.assert_allclose(result[:, 3], 0.5, atol=0.05)
