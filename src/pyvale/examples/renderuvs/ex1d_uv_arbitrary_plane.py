# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Render UVs: Projection onto an arbitrary plane
================================================================================

Here we embed a regular grid in a tilted three-dimensional plane and generate
UVs using the plane normal, origin, and preferred upward direction.
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
# 1. Create and tilt a regular verification grid
# ------------------------------------------------------------
display_coords, connectivity = rectangle_grid()
angle = np.deg2rad(35.0)
axis_u = np.array((np.cos(angle), 0.0, -np.sin(angle)))
axis_v = np.array((0.0, 1.0, 0.0))
tilted_coords = embed_grid(display_coords, axis_u, axis_v)

# %%
# 2. Define the arbitrary projection plane and generate UVs
# ------------------------------------------------------------
# Supplying ``up`` removes the rotational ambiguity around the plane normal
# and makes positive V follow the grid's vertical axis.
plane = render.UVPlane(
    normal=np.cross(axis_u, axis_v),
    origin=np.zeros(3),
    up=axis_v,
)
uvs = render.uv_project_planar_centered(
    tilted_coords,
    TEXTURE_SHAPE,
    span=0.9,
    plane=plane,
)

# %%
# 3. Render the arbitrary-plane mapping
# ------------------------------------------------------------
# We display the mapping face-on so its texture orientation can be inspected
# independently of camera perspective.
texture = render.image_load(dataset.riley_cal_target_texture_path())
output_dir = (
    Path.cwd() / "pyvale-output" / "renderuvs_ex1d_uv_arbitrary_plane"
)
render_uv_variant(
    display_coords, connectivity, uvs, texture, output_dir / "tilted",
)

print(f"Rendered the arbitrary-plane UV mapping to {output_dir}")

# %%
# The mapping generated from the 35-degree tilted plane is shown below.
#
# .. image:: ../../_static/renderuvs_ex1d_uv_arbitrary_plane.png
#    :alt: Calibration texture projected onto an arbitrary plane
#    :width: 500px
#    :align: center
