# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Smoke tests for the documented 2D image-warp gallery examples."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest


_RENDER2D_EXAMPLES = (
    pytest.param(
        "render2d/ex1a_pixint_grid_one_elem_types.py",
        "pyvale-output/render-pixint-grid-one-element/quad9.npy",
    ),
    pytest.param(
        "render2d/ex1b_pixint_grid_mesh.py",
        "pyvale-output/render-pixint-grid-mesh/warped_images.npy",
    ),
    pytest.param(
        "render2d/ex1c_pixint_speck_newton.py",
        "pyvale-output/render-pixint-speck/warped_images.npy",
        marks=pytest.mark.example_slow,
    ),
    pytest.param(
        "render2d/ex2a_imagedef2d_planar_deformation.py",
        "pyvale-output/render-imagedef2d/warped_images.npy",
    ),
)


@pytest.mark.example
@pytest.mark.parametrize(("example", "output"), _RENDER2D_EXAMPLES)
def test_render2d_example(
    run_example: Callable[..., Path],
    example: str,
    output: str,
) -> None:
    """Each 2D image-warp gallery example runs and writes its result."""
    run_example(example, (output,), timeout=300.0)
