# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Render UVs: Compare texture boundary modes
================================================================================

Here we map a three dimensional plate with hole mesh beyond a source texture's
pixel bounds. We compare a seamless periodic texture, a non periodic texture
whose repeated edges form a seam, and the default saturated boundary mode.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.data as dataset
import pyvale.dataio as io
from pyvale import render
from pyvale.examples.renderuvs.tools import render_uv_example

# %%
# 1. Load and orient the packaged plate with hole mesh
# ------------------------------------------------------------
# The native PyVale CSV loader provides the three dimensional surface mesh.
# Rotating about its centre makes the hole and exposed side edges legible.
data_dir = dataset.riley_platehole_csv_case_path()
simulation = io.MeshLoader(
    load_dir=data_dir,
    coords_file="coords.csv",
    connect_files="connect.csv",
    load_opts=io.SimLoadOpts(coord_header=None),
).load_mesh()

base_mesh = render.mesh3d_from_simdata(simulation, shader=None)

oriented_mesh = render.mesh_rotate(
    base_mesh,
    Rotation.from_euler("xyz", (0.0, 22.0, 8.0), degrees=True),
    pivot=render.mesh_center(base_mesh),
)

# %%
# 2. Build the experimental camera and calculate its image scale
# ----------------------------------------------------------------------
camera = render.Camera(
    pixels_num=np.array((1792, 1120)),
    pixels_size=np.array((5.5e-6, 5.5e-6)),
    pos_world=np.zeros(3),
    rot_world=Rotation.identity(),
    roi_cent_world=render.mesh_center(oriented_mesh),
    focal_length=35.0e-3,
    subsample=4,
)

camera = render.cam_frame_mesh(
    camera,
    oriented_mesh,
    fov_scale=render.cam_coverage_to_fov_scale(0.90),
)

image_leng_per_px = render.cam_calc_leng_per_px(camera)

# %%
# 3. Build periodic and non periodic source textures
# ----------------------------------------------------------------------
# A periodic checker repeats without a seam. A horizontal ramp is deliberately
# non periodic: its bright right edge meets its dark left edge when tiled, so
# the boundary is unmistakable in the rendered result.
tile_size = 128
tile_columns, tile_rows = np.meshgrid(
    np.arange(tile_size),
    np.arange(tile_size),
)
checker = ((tile_columns // 16 + tile_rows // 16) % 2) * 190 + 32
seamless_texture = checker.astype(np.uint8)
ramp = np.linspace(24, 232, 256, dtype=np.uint8)
nonperiodic_texture = np.tile(ramp, (tile_size, 1))

# %%
# 4. Define a physical feature size and cross the texture boundary
# ----------------------------------------------------------------------
# The checker has 16 texture pixels per square. Requesting 16 image pixels per
# square and using the camera scale defines its physical size. Moving the
# mapping centre close to the right edge guarantees that the plate crosses a
# boundary in both the tiled and saturated cases.
texture_px_per_leng = render.uv_calc_texture_px_per_leng_from_image(
    texture_px_per_feature=16.0,
    image_px_per_feature=16.0,
    image_leng_per_px=image_leng_per_px,
)
nonperiodic_height, nonperiodic_width = nonperiodic_texture.shape[:2]

seamless_mapping = render.uv_map_planar_scaled(
    oriented_mesh.coords,
    seamless_texture,
    texture_px_per_leng,
    plane=render.EUVPlane.XY,
    texture_center_px=np.array((0.90 * tile_size, 0.50 * tile_size)),
    bounds=render.EUVBounds.TILED,
)

seamed_mapping = render.uv_map_planar_scaled(
    oriented_mesh.coords,
    nonperiodic_texture,
    texture_px_per_leng,
    plane=render.EUVPlane.XY,
    texture_center_px=np.array(
        (0.90 * nonperiodic_width, 0.50 * nonperiodic_height)
    ),
    bounds=render.EUVBounds.TILED,
)

saturated_mapping = render.uv_map_planar_scaled(
    oriented_mesh.coords,
    nonperiodic_texture,
    texture_px_per_leng,
    plane=render.EUVPlane.XY,
    texture_center_px=np.array(
        (0.90 * nonperiodic_width, 0.50 * nonperiodic_height)
    ),
)

# %%
# 5. Render the three texture boundary cases
# ----------------------------------------------------------------------
output_dir = Path.cwd() / "pyvale-output" / "renderuvs_ex1c_uv_pixel_region"

for variant_name, mapping in (
    ("seamless", seamless_mapping),
    ("tiled_seam", seamed_mapping),
    ("saturated", saturated_mapping),
):
    textured_mesh = render.Mesh3D(
        element_type=oriented_mesh.element_type,
        coords=oriented_mesh.coords,
        connectivity=oriented_mesh.connectivity,
        shader=render.RileyTextureShader(
            uvs=mapping.uvs,
            texture=mapping.texture,
        ),
    )
    render_uv_example(textured_mesh, camera, output_dir / variant_name)

print(f"Rendered the texture boundary comparisons to {output_dir}")
print(
    f"Image scale={image_leng_per_px:.6g} length/px, "
    f"seamless tiles={seamless_mapping.tile_counts}, "
    f"non periodic tiles={seamed_mapping.tile_counts}"
)

# %%
# From left to right: a periodic texture tiles seamlessly, a non periodic source
# exposes its repeated edge, and saturation stretches the edge pixels wherever
# the requested mapping lies beyond the source image.
#
# .. image:: ../../_static/renderuvs_ex1c_uv_pixel_region.png
#    :alt: Seamless, visibly seamed, and saturated texture boundary modes
#    :width: 900px
#    :align: center
