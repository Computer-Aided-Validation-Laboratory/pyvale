# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Smoke tests for non-interactive documented DIC examples."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

import pyvale.render as render


_DIC_EXAMPLES = (
    pytest.param("dic/ex05_dic_challenge.py", "pyvale-output"),
    pytest.param(
        "dic/ex07_incremental.py",
        "pyvale-output/incremental",
        marks=pytest.mark.example_slow,
    ),
)


@pytest.mark.example
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
@pytest.mark.blender
@pytest.mark.skipif(
    not render.blender_available(),
    reason="The optional Blender renderer backend is unavailable.",
)
def test_blender_dic_example(run_example: Callable[..., Path]) -> None:
    """The Blender-to-DIC example runs when Blender is available."""
    run_example("dic/ex04_dic_blender.py", ("pyvale-output",), timeout=300.0)
