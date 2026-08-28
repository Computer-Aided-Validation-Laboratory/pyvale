# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Render UVs: Map into a texture pixel region
================================================================================

Here we place a three-dimensional plate-with-hole mesh into an explicitly
selected pixel rectangle in a speckle texture. The oblique physical view keeps
the hole and the plate thickness visible while the UV region is changed.
"""

from pathlib import Path

from scipy.spatial.transform import Rotation

import pyvale.data as dataset
import pyvale.dataio as io
from pyvale import render
from pyvale.examples.renderuvs.tools import render_uv_example

# %%
# 1. Load and orient the packaged plate-with-hole mesh
# ------------------------------------------------------------
# The native PyVale CSV loader provides the three-dimensional surface mesh.
# Rotating about its centre makes the hole and exposed side edges legible.
data_dir = dataset.riley_platehole_csv_case_path()
simulation = io.SimLoaderByField(
    load_dir=data_dir,
    coords_file="coords.csv",
    time_step_file=None,
    node_field_files=None,
    connect_files="connect.csv",
    load_opts=io.SimLoadOpts(coord_header=None),
).load_all_sim_data()
base_mesh = render.mesh3d_from_simdata(simulation, shader=None)
oriented_mesh = render.mesh_rotate(
    base_mesh,
    Rotation.from_euler("xyz", (0.0, 22.0, 8.0), degrees=True),
    pivot=render.mesh_center(base_mesh),
)

# %%
# 2. Generate UVs inside a specified pixel rectangle
# ------------------------------------------------------------
# The rectangle leaves a substantial texture border and ``CONTAIN`` keeps the
# whole plate visible without changing its physical aspect ratio.
texture = render.image_load(dataset.riley_speckle_texture_path())
height, width = texture.shape[:2]
pixel_bounds = (0.2 * width, 0.15 * height, 0.8 * width, 0.85 * height)
uvs = render.uv_project_planar_pixels(
    oriented_mesh.coords,
    texture.shape[:2],
    pixel_bounds,
    plane=render.EUVPlane.XY,
    fit=render.EUVFit.CONTAIN,
)
textured_mesh = render.Mesh3D(
    element_type=oriented_mesh.element_type,
    coords=oriented_mesh.coords,
    connectivity=oriented_mesh.connectivity,
    shader=render.RileyTextureShader(uvs=uvs, texture=texture),
)

# %%
# 3. Render the mapped pixel region
# ------------------------------------------------------------
output_dir = Path.cwd() / "pyvale-output" / "renderuvs_ex1c_uv_pixel_region"
camera_rotation = Rotation.identity()
render_uv_example(textured_mesh, output_dir / "region", camera_rotation)

print(f"Rendered the pixel-region UV mapping to {output_dir}")

# %%
# The three-dimensional plate is mapped into the selected speckle-image region.
#
# .. image:: ../../_static/renderuvs_ex1c_uv_pixel_region.png
#    :alt: Oblique plate with a hole mapped into a selected texture region
#    :width: 500px
#    :align: center
