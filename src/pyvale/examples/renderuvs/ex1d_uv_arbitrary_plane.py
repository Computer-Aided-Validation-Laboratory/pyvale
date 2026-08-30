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
texture_px_per_leng = render.uv_calc_texture_px_per_leng(
    texture_px_per_feature=177.1,
    feature_leng=1.25e-3,
)
mapping = render.uv_map_planar_scaled(
    oriented_mesh.coords,
    texture,
    texture_px_per_leng,
    plane=plane,
)
textured_mesh = render.Mesh3D(
    element_type=oriented_mesh.element_type,
    coords=oriented_mesh.coords,
    connectivity=oriented_mesh.connectivity,
    shader=render.RileyTextureShader(
        uvs=mapping.uvs,
        texture=mapping.texture,
    ),
)

# %%
# 3. Render the arbitrary-plane mapping
# ------------------------------------------------------------
output_dir = Path.cwd() / "pyvale-output" / "renderuvs_ex1d_uv_arbitrary_plane"
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
render_uv_example(textured_mesh, camera, output_dir / "tilted")

image_leng_per_px = render.cam_calc_leng_per_px(camera)
image_px_per_feature_pitch = render.uv_calc_image_px_per_feature(
    1.25e-3,
    image_leng_per_px,
)

print(f"Rendered the arbitrary-plane UV mapping to {output_dir}")
print(
    f"The 1.25 mm feature pitch is approximately "
    f"{image_px_per_feature_pitch:.2f} px at the ROI"
)

# %%
# The tilted calibration plate is mapped using its rotated local surface plane.
#
# .. image:: ../../_static/renderuvs_ex1d_uv_arbitrary_plane.png
#    :alt: Oblique three-dimensional calibration plate with arbitrary-plane UVs
#    :width: 500px
#    :align: center
