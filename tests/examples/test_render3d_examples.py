# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Smoke tests for the documented 3D render gallery examples."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

import pyvale.render as render


_RILEY_EXAMPLES = (
    pytest.param(
        "render3d/ex1a_riley_quickstart.py",
        "pyvale-output/render-riley-quickstart/cam0_frame0_field0.bmp",
    ),
)

_BLENDER_EXAMPLES = (
    "render3d/ex2a_blender_scene.py",
    "render3d/ex2b_blender_deformation.py",
    "render3d/ex2c_blender_stereo.py",
    "render3d/ex2d_blender_calibration.py",
    "render3d/ex2e_blender_stereo_deformation.py",
    "render3d/ex2f_blender_calibration_target.py",
)


@pytest.mark.example
@pytest.mark.parametrize(("example", "output"), _RILEY_EXAMPLES)
def test_render3d_example(
    run_example: Callable[..., Path],
    example: str,
    output: str,
) -> None:
    """Each non-Blender 3D render gallery example runs and writes its result."""
    run_example(example, (output,), timeout=300.0)


@pytest.mark.example
@pytest.mark.blender
@pytest.mark.skipif(
    not render.blender_available(),
    reason="The optional Blender renderer backend is unavailable.",
)
@pytest.mark.parametrize("example", _BLENDER_EXAMPLES)
def test_blender_render3d_example(
    run_example: Callable[..., Path],
    example: str,
) -> None:
    """Run each Blender gallery example when the optional backend is present."""
    work_dir = run_example(example, ("pyvale-output",), timeout=300.0)

    if example.endswith("ex2f_blender_calibration_target.py"):
        images = tuple((
            work_dir
            / "pyvale-output"
            / "render-blender-calibration-images"
            / "calimages"
        ).glob("*.tiff"))
        assert len(images) == 10
