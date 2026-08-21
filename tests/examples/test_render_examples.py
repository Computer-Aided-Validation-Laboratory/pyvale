# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Smoke tests for the documented render gallery examples."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

import pyvale.render as render


_RENDER_EXAMPLES = (
    pytest.param(
        "render/ex1a_riley_quickstart.py",
        "pyvale-output/render-riley-quickstart/cam0_frame0_field0.bmp",
    ),
    pytest.param(
        "render/ex3_imagedef2d_planar_deformation.py",
        "pyvale-output/render-imagedef2d/warped_images.npy",
    ),
    pytest.param(
        "render/ex4_pixint_grid_one_elem_types.py",
        "pyvale-output/render-pixint-grid-one-element/quad9.npy",
    ),
    pytest.param(
        "render/ex4a_pixint_grid_mesh.py",
        "pyvale-output/render-pixint-grid-mesh/warped_images.npy",
    ),
    pytest.param(
        "render/ex5_pixint_speck_newton.py",
        "pyvale-output/render-pixint-speck/warped_images.npy",
        marks=pytest.mark.example_slow,
    ),
)

_BLENDER_EXAMPLES = (
    "render/ex2a_blender_scene.py",
    "render/ex2b_blender_deformation.py",
    "render/ex2c_blender_stereo.py",
    "render/ex2d_blender_calibration.py",
    "render/ex2e_blender_stereo_deformation.py",
    "render/ex2f_blender_calibration_target.py",
)


@pytest.mark.example
@pytest.mark.parametrize(("example", "output"), _RENDER_EXAMPLES)
def test_render_example(
    run_example: Callable[..., Path],
    example: str,
    output: str,
) -> None:
    """Each non-Blender render gallery example runs and writes its result."""
    run_example(example, (output,), timeout=300.0)


@pytest.mark.example
@pytest.mark.blender
@pytest.mark.skipif(
    not render.blender_available(),
    reason="The optional Blender renderer backend is unavailable.",
)
@pytest.mark.parametrize("example", _BLENDER_EXAMPLES)
def test_blender_render_example(
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
