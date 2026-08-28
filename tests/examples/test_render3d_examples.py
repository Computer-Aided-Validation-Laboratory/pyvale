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
        "pyvale-output/render3d_ex1a_riley_quickstart/"
        "cam0_frame0_field0.bmp",
    ),
    pytest.param(
        "render3d/ex1b_riley_sphere200.py",
        "pyvale-output/render3d_ex1b_riley_sphere200/"
        "cam0_frame0_field0.bmp",
    ),
    pytest.param(
        "render3d/ex1c_riley_psf.py",
        "pyvale-output/render3d_ex1c_riley_psf/global_subpx_full/"
        "cam0_frame0_field0.bmp",
    ),
    pytest.param(
        "render3d/ex1d_riley_rabbits.py",
        "pyvale-output/render3d_ex1d_riley_rabbits/"
        "cam0_frame0_field0.bmp",
    ),
    pytest.param(
        "render3d/ex1e_riley_dicuq.py",
        "pyvale-output/render3d_ex1e_riley_dicuq/cam0_frame0_field0.bmp",
    ),
    pytest.param(
        "render3d/ex1f_riley_dic_from_exodus.py",
        "pyvale-output/render3d_ex1f_riley_dic_from_exodus/"
        "cam0_frame0_field0.bmp",
    ),
    pytest.param(
        "render3d/ex1g_riley_stereocal.py",
        "pyvale-output/render3d_ex1g_riley_stereocal/"
        "cam0_frame0_field0.bmp",
    ),
    pytest.param(
        "render3d/ex2d_blender_calibration.py",
        "pyvale-output/render3d_ex2d_blender_calibration/calibration/"
        "calibration.yaml",
    ),
)

_BLENDER_EXAMPLES = (
    pytest.param(
        "render3d/ex2a_blender_scene.py",
        "pyvale-output/render3d_ex2a_blender_scene/images/"
        "blenderimage_0.tiff",
    ),
    pytest.param(
        "render3d/ex2b_blender_deformation.py",
        "pyvale-output/render3d_ex2b_blender_deformation/images/"
        "blenderimage_0.tiff",
    ),
    pytest.param(
        "render3d/ex2c_blender_stereo.py",
        "pyvale-output/render3d_ex2c_blender_stereo/images/"
        "blenderimage_0_1.tiff",
    ),
    pytest.param(
        "render3d/ex2e_blender_stereo_deformation.py",
        "pyvale-output/render3d_ex2e_blender_stereo_deformation/images/"
        "blenderimage_0_1.tiff",
    ),
    pytest.param(
        "render3d/ex2f_blender_calibration_target.py",
        "pyvale-output/render3d_ex2f_blender_calibration_target/"
        "calimages/blendercal_1_1.tiff",
    ),
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
@pytest.mark.parametrize(("example", "output"), _BLENDER_EXAMPLES)
def test_blender_render3d_example(
    run_example: Callable[..., Path],
    example: str,
    output: str,
) -> None:
    """Run each Blender gallery example when the optional backend is present."""
    work_dir = run_example(example, (output,), timeout=300.0)

    if example.endswith("ex2f_blender_calibration_target.py"):
        images = tuple(
            (
                work_dir
                / "pyvale-output"
                / "render3d_ex2f_blender_calibration_target"
                / "calimages"
            ).glob("*.tiff")
        )
        assert len(images) == 10
