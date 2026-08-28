# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Render UVs: Projection onto an arbitrary plane
================================================================================

This example rotates the packaged three-dimensional calibration plate and
generates UVs from its rotated local surface plane. The plate thickness and an
oblique camera view make the arbitrary physical orientation clear.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.data as dataset
import pyvale.dataio as io
from pyvale import render
from pyvale.examples.renderuvs.tools import render_uv_example

# %%
# 1. Load and tilt the three-dimensional calibration plate
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
plate_rotation = Rotation.from_euler("xyz", (28.0, -34.0, 12.0), degrees=True)
oriented_mesh = render.mesh_rotate(
    base_mesh,
    plate_rotation,
    pivot=render.mesh_center(base_mesh),
)

# %%
# 2. Define the rotated surface plane and generate UV coordinates
# ------------------------------------------------------------
# The original plate face lies in XY. Rotating its normal and local V axis
# provides the arbitrary-plane definition without manually deriving a basis.
plane = render.UVPlane(
    normal=plate_rotation.apply(np.array((0.0, 0.0, 1.0))),
    origin=render.mesh_center(oriented_mesh),
    up=plate_rotation.apply(np.array((0.0, 1.0, 0.0))),
)
texture = render.image_load(dataset.riley_cal_target_texture_path())
uvs = render.uv_project_planar_centered(
    oriented_mesh.coords,
    texture.shape[:2],
    span=0.9,
    plane=plane,
)
textured_mesh = render.Mesh3D(
    element_type=oriented_mesh.element_type,
    coords=oriented_mesh.coords,
    connectivity=oriented_mesh.connectivity,
    shader=render.RileyTextureShader(uvs=uvs, texture=texture),
)

# %%
# 3. Render the arbitrary-plane mapping
# ------------------------------------------------------------
output_dir = Path.cwd() / "pyvale-output" / "renderuvs_ex1d_uv_arbitrary_plane"
render_uv_example(textured_mesh, output_dir / "tilted", Rotation.identity())

print(f"Rendered the arbitrary-plane UV mapping to {output_dir}")

# %%
# The tilted calibration plate is mapped using its rotated local surface plane.
#
# .. image:: ../../_static/renderuvs_ex1d_uv_arbitrary_plane.png
#    :alt: Oblique three-dimensional calibration plate with arbitrary-plane UVs
#    :width: 500px
#    :align: center
