# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Render UVs: Texture aspect and fit modes
================================================================================

This example maps the same oblique three-dimensional calibration plate with
each planar fit mode. The visible plate edges keep the camera view constant
while the texture fitting behaviour changes.
"""

from pathlib import Path

from scipy.spatial.transform import Rotation

import pyvale.data as dataset
import pyvale.dataio as io
from pyvale import render
from pyvale.examples.renderuvs.tools import render_uv_example

# %%
# 1. Load the packaged three-dimensional calibration plate
# ------------------------------------------------------------
data_dir = dataset.riley_stereocal_case_path()
simulation = io.SimLoaderByField(
    load_dir=data_dir,
    coords_file="coords.csv",
    time_step_file=None,
    node_field_files=None,
    connect_files="connect.csv",
    load_opts=io.SimLoadOpts(coord_header=None),
).load_all_sim_data()
base_mesh = render.mesh3d_from_simdata(simulation, shader=None)
texture = render.image_load(dataset.riley_cal_target_texture_path())

# %%
# 2. Generate UVs using each fit mode
# ------------------------------------------------------------
# The plate remains in one oblique orientation. ``CONTAIN`` preserves the full
# projection, fit-U and fit-V select one fitted texture axis, and ``STRETCH``
# fills both axes independently.
fit_modes = (
    ("contain", render.EUVFit.CONTAIN),
    ("fit_u", render.EUVFit.FIT_U),
    ("fit_v", render.EUVFit.FIT_V),
    ("stretch", render.EUVFit.STRETCH),
)
mesh_rotation = Rotation.from_euler("xyz", (0.0, 24.0, 8.0), degrees=True)
camera_rotation = Rotation.from_euler("xyz", (16.0, -24.0, 0.0), degrees=True)
oriented_mesh = render.mesh_rotate(
    base_mesh,
    mesh_rotation,
    pivot=render.mesh_center(base_mesh),
)

# %%
# 3. Render and compare the mappings
# ------------------------------------------------------------
output_dir = Path.cwd() / "pyvale-output" / "renderuvs_ex1b_uv_texture_aspect"

for variant_name, fit_mode in fit_modes:
    uvs = render.uv_project_planar(
        oriented_mesh.coords,
        texture_shape=texture.shape[:2],
        fit=fit_mode,
    )
    textured_mesh = render.Mesh3D(
        element_type=oriented_mesh.element_type,
        coords=oriented_mesh.coords,
        connectivity=oriented_mesh.connectivity,
        shader=render.RileyTextureShader(uvs=uvs, texture=texture),
    )
    render_uv_example(textured_mesh, output_dir / variant_name, camera_rotation)

print(f"Rendered UV fit variants to {output_dir}")

# %%
# Contain, fit-U, fit-V, and stretch are shown from left to right below.
#
# .. image:: ../../_static/renderuvs_ex1b_uv_texture_aspect.png
#    :alt: Four planar UV texture fitting modes on a three-dimensional plate
#    :width: 1000px
#    :align: center
