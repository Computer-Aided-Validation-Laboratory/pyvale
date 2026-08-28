# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Render UVs: Translate, rotate, and scale a mapping
================================================================================

Here we retain the asymmetric Riley rabbit mesh, generate a centred mapping,
and apply a combined UV transformation. The physical rabbit is rendered from
an oblique view so its three-dimensional form remains visible.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.data as dataset
import pyvale.dataio as io
from pyvale import render
from pyvale.examples.renderuvs.tools import render_uv_example

# %%
# 1. Load and orient the packaged Riley rabbit mesh
# ------------------------------------------------------------
# The asymmetric rabbit makes UV rotations and translations easier to recognise
# than a rectangular plate, so it remains the best mesh for this comparison.
data_dir = dataset.riley_rabbit_case_path("riley", render.EElementType.QUAD4)
simulation = io.SimLoaderByField(
    load_dir=data_dir,
    coords_file="coords.csv",
    time_step_file=None,
    node_field_files=None,
    connect_files="connectivity.csv",
    load_opts=io.SimLoadOpts(coord_header=None),
).load_all_sim_data()
base_mesh = render.mesh3d_from_simdata(simulation, shader=None)
oriented_mesh = render.mesh_rotate(
    base_mesh,
    Rotation.from_euler("xyz", (8.0, -24.0, 0.0), degrees=True),
    pivot=render.mesh_center(base_mesh),
)
texture = render.image_load(dataset.riley_cal_target_texture_path())

# %%
# 2. Generate the original UV coordinates
# ------------------------------------------------------------
original_uvs = render.uv_project_planar_centered(
    oriented_mesh.coords,
    texture.shape[:2],
    span=0.75,
)

# %%
# 3. Apply a combined transformation
# ------------------------------------------------------------
# Scaling and rotation happen about the selected pivot. Translation is applied
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
camera_rotation = Rotation.identity()

for variant_name, uvs in (
    ("original", original_uvs),
    ("transformed", transformed_uvs),
):
    textured_mesh = render.Mesh3D(
        element_type=oriented_mesh.element_type,
        coords=oriented_mesh.coords,
        connectivity=oriented_mesh.connectivity,
        shader=render.RileyTextureShader(uvs=uvs, texture=texture),
    )
    render_uv_example(textured_mesh, output_dir / variant_name, camera_rotation)

print(f"Rendered the original and combined UV transform to {output_dir}")

# %%
# The original mapping is on the left and the combined transformed variant is
# on the right.
#
# .. image:: ../../_static/renderuvs_ex1e_uv_transform.png
#    :alt: Original and transformed UV mappings on a three-dimensional rabbit
#    :width: 900px
#    :align: center
