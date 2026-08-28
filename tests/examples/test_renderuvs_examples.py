# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ============================================================================

"""Smoke tests for the documented Render UV gallery examples."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


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
    """Each representative render is visible, centred, and uncropped."""
    work_dir = run_example(example, (output,), timeout=300.0)
    with Image.open(work_dir / output) as output_image:
        image = np.asarray(output_image)

    background = image[0, 0]
    foreground = np.any(image != background, axis=2)
    rows, columns = np.nonzero(foreground)

    assert rows.size > 0
    assert columns.min() > 0
    assert rows.min() > 0
    assert columns.max() < image.shape[1] - 1
    assert rows.max() < image.shape[0] - 1

    object_center = np.array(
        (
            0.5 * (columns.min() + columns.max()),
            0.5 * (rows.min() + rows.max()),
        )
    )
    image_center = 0.5 * np.array((image.shape[1] - 1, image.shape[0] - 1))
    assert np.all(np.abs(object_center - image_center) < 0.1 * image_center)
