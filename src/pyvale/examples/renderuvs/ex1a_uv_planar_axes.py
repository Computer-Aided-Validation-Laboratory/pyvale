# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Render UVs: Axis-aligned planar projection
================================================================================

This example maps a textured, three-dimensional calibration plate after
orienting it in the XY, YZ, and XZ planes. The plate thickness and oblique
camera views make each physical orientation visible in the rendered result.
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
# The calibration plate is a TRI3 surface mesh with front, back, and side
# faces. Its exposed edges make planar-projection orientation easy to inspect.

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

mesh_center = render.mesh_center(base_mesh)

# %%
# 2. Orient the plate in each axis-aligned projection plane
# ------------------------------------------------------------
# Each rotation moves the broad plate face into the requested world plane.
# UVs are then generated from the same plane, while an oblique camera reveals
# the plate thickness and its physical orientation.
variants = (
    (
        "xy",
        render.EUVPlane.XY,
        Rotation.identity(),
        Rotation.from_euler("y", 10.0, degrees=True),
    ),
    (
        "yz",
        render.EUVPlane.YZ,
        Rotation.from_euler("y", 88.0, degrees=True),
        Rotation.from_euler("y", 78.0, degrees=True),
    ),
    (
        "xz",
        render.EUVPlane.XZ,
        Rotation.from_euler("x", 80.0, degrees=True),
        Rotation.from_euler("x", 70.0, degrees=True),
    ),
)

# %%
# 3. Generate and render the three mappings
# ------------------------------------------------------------
# ``uv_project_planar_centered`` maps the selected physical plane into a
# centred texture region, leaving a five-percent border on every side.
output_dir = Path.cwd() / "pyvale-output" / "renderuvs_ex1a_uv_planar_axes"

for variant_name, projection_plane, mesh_rotation, camera_rotation in variants:
    oriented_mesh = render.mesh_rotate(
        base_mesh,
        mesh_rotation,
        pivot=mesh_center,
    )
    uvs = render.uv_project_planar_centered(
        oriented_mesh.coords,
        texture.shape[:2],
        span=0.9,
        plane=projection_plane,
    )
    textured_mesh = render.Mesh3D(
        element_type=oriented_mesh.element_type,
        coords=oriented_mesh.coords,
        connectivity=oriented_mesh.connectivity,
        shader=render.RileyTextureShader(uvs=uvs, texture=texture),
    )
    render_uv_example(textured_mesh, output_dir / variant_name, camera_rotation)

print(f"Rendered axis-aligned UV projections to {output_dir}")

# %%
# The XY, YZ, and XZ variants are shown from left to right below.
#
# .. image:: ../../_static/renderuvs_ex1a_uv_planar_axes.png
#    :alt: Three-dimensional calibration plate with XY, YZ, and XZ UV maps
#    :width: 900px
#    :align: center
