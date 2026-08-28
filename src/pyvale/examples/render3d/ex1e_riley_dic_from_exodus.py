# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Riley: DIC UQ from Exodus
================================================================================
Here we render the same plate with a hole in tension from the last example but 
we load the data from an exodus file and generate the uv's using Riley's tools.
"""

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import riley

import pyvale.data as dataset
from pyvale import render
from pyvale.mooseherder import ExodusLoader

# %%
# 1. Load Exodus data and build a textured surface mesh
# ------------------------------------------------------------
simulation = ExodusLoader(
    dataset.riley_platehole_exodus_path(),
    enforce_convention=True,
).load_all_sim_data()

texture = riley.load_texture_u8(dataset.riley_speckle_texture_path())

surface_mesh = render.mesh3d_from_simdata(
    simulation,
    shader=None,
    displacement_keys=("disp_x", "disp_y", "disp_z"),
)

surface_mesh.shader = render.RileyTextureShader(
    uvs=riley.project_uvs_planar_centered(
        surface_mesh.coords,
        (2464, 2056),
        uv_span_max=0.8,
        proj_plane=(
            np.array((0.0, 0.0, -1.0), dtype=np.float64),
            np.array((0.0, 0.0, 0.0), dtype=np.float64),
        ),
    ),
    texture=texture,
)

# To save time we only render the first and last frame. If you want to render
# all frames comment this out.
surface_mesh.displacements = surface_mesh.displacements[[0, -1]]

# %%
# 2. Create and position a distorted stereo camera pair
# ------------------------------------------------------------
# The cameras are built natively so they can be saved in Riley's format without 
# any conversion of their parameters.
pixels_num = (2464, 2056)
pixels_size = (3.45e-6, 3.45e-6)
focal_length = 50.0e-3
roi_centre = riley.roi_cent_from_coords(surface_mesh.coords)

camera_0 = riley.Camera(
    pixels_num=pixels_num,
    pixels_size=pixels_size,
    pos_world=riley.pos_fill_frame_from_rot(
        surface_mesh.coords,
        pixels_num,
        pixels_size,
        focal_length,
        (0.0, 0.0, 0.0),
        0.65,
    ),
    rot_world=(0.0, 0.0, 0.0),
    roi_cent_world=roi_centre,
    focal_length=focal_length,
    sub_sample=2,
    distortion_model=int(render.EDistortionModel.BROWN_CONRADY),
    distortion_k1=-0.2,
    distortion_k2=0.1,
    distortion_p1=0.0001,
    distortion_p2=-0.0001,
)

camera_1 = copy.deepcopy(camera_0)
camera_1.rot_world = (0.0, float(np.deg2rad(20.0)), 0.0)
camera_1.pos_world = riley.pos_fill_frame_from_rot(
    surface_mesh.coords,
    pixels_num,
    pixels_size,
    focal_length,
    camera_1.rot_world,
    0.65,
)

cameras = [camera_0, camera_1]

# %%
# 3. Configure and build the renderer
# ------------------------------------------------------------

config = riley.create_raster_config(
    num_frames=surface_mesh.displacements.shape[0],
    total_threads=8,
    save_strategy=riley.SaveStrategy.disk,
)

config.background_value = 128.0
config.tile_size_max = 128
config.save_scaling = riley.ScaleStrategy.none

output_dir = (
    Path.cwd() / "pyvale-output"
    / "render3d_ex1e_riley_dic_from_exodus"
)

renderer = render.Riley(config, output_dir)

# %%
# 4. Build the scene and render the extracted surface
# ------------------------------------------------------------
result = renderer.render(
    render.Scene3D(meshes=[surface_mesh], cameras=cameras)
)

# %%
# 5. Save the stereo pair in Riley's exchange format
# ------------------------------------------------------------
riley.save_stereo_pair(
    str(output_dir),
    "stereo_data_opengl.csv",
    cameras[0],
    cameras[1],
)

riley.save_stereo_pair(
    str(output_dir),
    "stereo_data_opencv.csv",
    replace(cameras[0], coord_sys=riley.CameraCoordSys.opencv),
    replace(cameras[1], coord_sys=riley.CameraCoordSys.opencv),
)

print(f"Rendered Exodus-driven DIC images to {output_dir}")
print(f"{result.images=}")

# %%
# The first frame from both cameras is combined side by side below.
#
# .. image:: ../../../../_static/render3d_ex1e_riley_dic_from_exodus.png
#    :alt: Riley stereo DIC render loaded from Exodus data
#    :width: 900px
#    :align: center
