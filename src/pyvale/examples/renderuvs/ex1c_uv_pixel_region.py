# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Render UVs: Map into a texture pixel region
================================================================================

Here we place a packaged plate-with-hole mesh inside an explicitly selected
pixel rectangle in the speckle texture. Pixel bounds are useful when the
specimen location in an experimental texture is already known.
"""

from pathlib import Path

import riley

import pyvale.data as dataset
import pyvale.dataio as io
from pyvale import render

from pyvale.examples._renderuv_tools import render_uv_variant

# %%
# 1. Load the packaged plate-with-hole mesh
# ------------------------------------------------------------
# We use the native PyVale CSV loader and keep only the static reference mesh
# needed to explain UV placement.
data_dir = dataset.riley_platehole_csv_case_path()
simulation = io.SimLoaderByField(
    load_dir=data_dir,
    coords_file="coords.csv",
    time_step_file=None,
    node_field_files=None,
    connect_files="connect.csv",
    load_opts=io.SimLoadOpts(coord_header=None),
).load_all_sim_data()
mesh = render.mesh3d_from_simdata(simulation, shader=None)

# %%
# 2. Generate UVs inside a specified pixel rectangle
# ------------------------------------------------------------
# The rectangle leaves a substantial texture border and ``CONTAIN`` keeps the
# whole plate visible without changing its physical aspect ratio.
texture = render.image_load(dataset.riley_speckle_texture_path())
texture_shape = texture.shape[:2]
height, width = texture_shape
pixel_bounds = (0.2 * width, 0.15 * height, 0.8 * width, 0.85 * height)
uvs = render.project_uvs_planar_pixels(
    mesh.coords,
    texture_shape,
    pixel_bounds,
    plane=render.EUVPlane.XY,
    fit=render.EUVFit.CONTAIN,
)

# %%
# 3. Render the mapped pixel region
# ------------------------------------------------------------
output_dir = (
    Path.cwd() / "pyvale-output" / "renderuvs_ex1c_uv_pixel_region"
)
render_uv_variant(
    mesh.coords, mesh.connectivity, uvs, texture, output_dir / "region",
)

print(f"Rendered the pixel-region UV mapping to {output_dir}")

# %%
# The plate mapped into the selected speckle-image region is shown below.
#
# .. image:: ../../_static/renderuvs_ex1c_uv_pixel_region.png
#    :alt: Plate with a hole mapped into a selected texture pixel region
#    :width: 500px
#    :align: center
