# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ============================================================================

"""Smoke tests for the documented Render UV gallery examples."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest


_RENDER_UV_EXAMPLES = (
    pytest.param(
        "renderuvs/ex1a_uv_planar_axes.py",
        "pyvale-output/renderuvs_ex1a_uv_planar_axes/xy/"
        "cam0_frame0_field0.bmp",
    ),
    pytest.param(
        "renderuvs/ex1b_uv_texture_aspect.py",
        "pyvale-output/renderuvs_ex1b_uv_texture_aspect/contain/"
        "cam0_frame0_field0.bmp",
    ),
    pytest.param(
        "renderuvs/ex1c_uv_pixel_region.py",
        "pyvale-output/renderuvs_ex1c_uv_pixel_region/region/"
        "cam0_frame0_field0.bmp",
    ),
    pytest.param(
        "renderuvs/ex1d_uv_arbitrary_plane.py",
        "pyvale-output/renderuvs_ex1d_uv_arbitrary_plane/tilted/"
        "cam0_frame0_field0.bmp",
    ),
    pytest.param(
        "renderuvs/ex1e_uv_transform.py",
        "pyvale-output/renderuvs_ex1e_uv_transform/transformed/"
        "cam0_frame0_field0.bmp",
    ),
)


@pytest.mark.example
@pytest.mark.parametrize(("example", "output"), _RENDER_UV_EXAMPLES)
def test_renderuv_example(
    run_example: Callable[..., Path],
    example: str,
    output: str,
) -> None:
    """Each Render UV example runs and writes its representative result."""
    run_example(example, (output,), timeout=300.0)
