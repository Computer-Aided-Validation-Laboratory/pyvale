# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Render UVs: Axis-aligned planar projection
================================================================================

Here we generate UV coordinates by projecting the same regular grid in the
XY, YZ, and XZ planes. Riley then renders each result with a calibration
texture so that the texture orientation is easy to inspect.
"""

from pathlib import Path

import numpy as np
import riley

import pyvale.data as dataset
from pyvale import render

from pyvale.examples._renderuv_tools import (
    TEXTURE_SHAPE,
    embed_grid,
    rectangle_grid,
    render_uv_variant,
)

# %%
# 1. Create regular grids in the three axis-aligned planes
# ------------------------------------------------------------
# The verification module creates one regular Quad4 grid. We embed copies in
# each three-dimensional plane, avoiding any additional packaged mesh data.
display_coords, connectivity = rectangle_grid()
plane_coords = {
    "xy": embed_grid(display_coords, np.array((1, 0, 0)),
                     np.array((0, 1, 0))),
    "yz": embed_grid(display_coords, np.array((0, 1, 0)),
                     np.array((0, 0, 1))),
    "xz": embed_grid(display_coords, np.array((1, 0, 0)),
                     np.array((0, 0, 1))),
}
planes = {
    "xy": render.EUVPlane.XY,
    "yz": render.EUVPlane.YZ,
    "xz": render.EUVPlane.XZ,
}

# %%
# 2. Generate centred UV coordinates for each plane
# ------------------------------------------------------------
# The texture shape uses NumPy order, ``(height, width)``. A span of 0.9 leaves
# a five-percent border around the projected grid.
uv_sets = {
    name: render.project_uvs_planar_centered(
        coords,
        TEXTURE_SHAPE,
        span=0.9,
        plane=planes[name],
    )
    for name, coords in plane_coords.items()
}

# %%
# 3. Render the three mappings with Riley
# ------------------------------------------------------------
# All three UV sets are rendered on the same face-on display grid. This makes
# the texture results directly comparable while the UVs themselves were
# generated from three differently embedded grids.
texture = render.image_load(dataset.riley_cal_target_texture_path())
output_dir = (
    Path.cwd() / "pyvale-output" / "renderuvs_ex1a_uv_planar_axes"
)
for name, uvs in uv_sets.items():
    render_uv_variant(
        display_coords, connectivity, uvs, texture, output_dir / name,
    )

print(f"Rendered axis-aligned UV projections to {output_dir}")

# %%
# The XY, YZ, and XZ variants are shown from left to right below.
#
# .. image:: ../../_static/renderuvs_ex1a_uv_planar_axes.png
#    :alt: Calibration texture mapped using XY, YZ, and XZ planar UVs
#    :width: 900px
#    :align: center
