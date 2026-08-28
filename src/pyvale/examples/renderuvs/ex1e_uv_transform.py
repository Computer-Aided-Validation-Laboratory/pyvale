# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Render UVs: Translate, rotate, and scale a mapping
================================================================================

Here we generate a centred mapping and apply a combined UV transformation.
The transform scales and rotates about a selected pivot before applying its
final translation.
"""

from pathlib import Path

import riley

import pyvale.data as dataset
import pyvale.dataio as io
from pyvale import render

from pyvale.examples._renderuv_tools import render_uv_variant

# %%
# 1. Load a packaged Riley rabbit mesh
# ------------------------------------------------------------
# The asymmetric rabbit silhouette makes rotations and translations easier to
# recognize than they would be on a rectangular grid.

data_dir = dataset.riley_rabbit_case_path(
    "riley",
    render.EElementType.QUAD4,
)

simulation = io.SimLoaderByField(
    load_dir=data_dir,
    coords_file="coords.csv",
    time_step_file=None,
    node_field_files=None,
    connect_files="connectivity.csv",
    load_opts=io.SimLoadOpts(coord_header=None),
).load_all_sim_data()

mesh = render.mesh3d_from_simdata(simulation, shader=None)
texture = render.image_load(dataset.riley_cal_target_texture_path())

# %%
# 2. Generate the original UV coordinates
# ------------------------------------------------------------
original_uvs = render.uv_project_planar_centered(
    mesh.coords,
    texture.shape[:2],
    span=0.75,
)

# %%
# 3. Apply a combined transformation
# ------------------------------------------------------------
# Scaling and rotation happen about the chosen pivot. Translation is applied
# last, and transformed UVs are allowed to extend outside the texture bounds.

transform = render.UVTransform(
    translation=(0.08, -0.04),
    rotation_degrees=18.0,
    scale=(0.85, 0.85),
    pivot=(0.5, 0.5),
)

transformed_uvs = render.uv_transform(original_uvs, transform)

# %%
# 4. Render the original and transformed mappings
# ------------------------------------------------------------
output_dir = Path.cwd() / "pyvale-output" / "renderuvs_ex1e_uv_transform"

render_uv_variant(
    mesh.coords, mesh.connectivity, original_uvs, texture,
    output_dir / "original",
)

render_uv_variant(
    mesh.coords, mesh.connectivity, transformed_uvs, texture,
    output_dir / "transformed",
)

print(f"Rendered the original and combined UV transform to {output_dir}")

# %%
# The original mapping is on the left and the combined transformed variant is
# on the right.
#
# .. image:: ../../_static/renderuvs_ex1e_uv_transform.png
#    :alt: Original and transformed rabbit UV mappings
#    :width: 900px
#    :align: center
