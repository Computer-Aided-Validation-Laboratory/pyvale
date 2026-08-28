# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Render UVs: Texture aspect and fit modes
================================================================================

Here we compare the four planar fit modes. Contain preserves the complete
projection, the U and V modes select one fitted axis, and stretch fills both
texture axes independently.
"""

from pathlib import Path

import riley

import pyvale.data as dataset
from pyvale import render

from pyvale.examples._renderuv_tools import (
    TEXTURE_SHAPE,
    rectangle_grid,
    render_uv_variant,
)

# %%
# 1. Create a wide regular grid
# ------------------------------------------------------------
# A wide grid and a differently proportioned texture make the behavior of each
# fit mode visible in the rendered calibration markings.
coords, connectivity = rectangle_grid(length_u=2.5, length_v=1.0)

# %%
# 2. Generate UVs using each fit mode
# ------------------------------------------------------------
fit_modes = {
    "contain": render.EUVFit.CONTAIN,
    "fit_u": render.EUVFit.FIT_U,
    "fit_v": render.EUVFit.FIT_V,
    "stretch": render.EUVFit.STRETCH,
}
uv_sets = {
    name: render.project_uvs_planar(
        coords,
        texture_shape=TEXTURE_SHAPE,
        fit=fit,
    )
    for name, fit in fit_modes.items()
}

# %%
# 3. Render and compare the mappings
# ------------------------------------------------------------
# ``FIT_V`` can extend beyond the U texture bounds in this example, while
# ``STRETCH`` deliberately changes the mapping aspect ratio.
texture = riley.load_texture_u8(dataset.riley_cal_target_texture_path())
output_dir = (
    Path.cwd() / "pyvale-output" / "renderuvs_ex1b_uv_texture_aspect"
)
for name, uvs in uv_sets.items():
    render_uv_variant(coords, connectivity, uvs, texture, output_dir / name)

print(f"Rendered UV fit variants to {output_dir}")

# %%
# Contain, fit-U, fit-V, and stretch are shown from left to right below.
#
# .. image:: ../../_static/renderuvs_ex1b_uv_texture_aspect.png
#    :alt: Four planar UV texture fitting modes
#    :width: 1000px
#    :align: center
