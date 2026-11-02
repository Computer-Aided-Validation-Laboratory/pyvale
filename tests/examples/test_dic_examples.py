# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Smoke tests for automated documented DIC examples."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

_DIC_EXAMPLES = (
    pytest.param("dic/ex04_dic_challenge.py", "pyvale-output/dic_ex04"),
    pytest.param(
        "dic/ex06_incremental.py",
        "pyvale-output/dic_ex06",
        marks=pytest.mark.example_slow,
    ),
    pytest.param("dic/ex08_stereo.py", "pyvale-output/dic_ex08"),
    pytest.param("dic/ex09_stereo_platehole.py", "pyvale-output/dic_ex09"),
    pytest.param("dic/ex10_dic_chal.py", "pyvale-output/dic_ex10"),
)


@pytest.mark.example
@pytest.mark.example_module("dic")
def test_plate_with_hole_examples(run_example: Callable[..., Path]) -> None:
    """The strain example runs against DIC output from the preceding example."""
    work_dir = run_example(
        "dic/ex02_plate_with_hole.py",
        ("pyvale-output/dic_ex02",),
        timeout=300.0,
    )

    run_example(
        "dic/ex03_plate_with_hole_strain.py",
        ("pyvale-output/dic_ex03",),
        timeout=300.0,
    )

    assert (work_dir / "pyvale-output/dic_ex03").is_dir()


@pytest.mark.example
@pytest.mark.parametrize(("example", "output"), _DIC_EXAMPLES)
def test_dic_example(
    run_example: Callable[..., Path],
    example: str,
    output: str,
) -> None:
    """Each supported standalone DIC gallery example runs successfully."""
    run_example(example, (output,), timeout=300.0)
