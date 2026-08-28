# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Riley: Digital Image Correlation UQ
================================================================================
In this example we render stereo camera images of a speckle pattern applued to
a plate with a hole loaded in tension. For this case we specifically choose 
parameters representative of typical stereo DIC setups.
"""

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import riley
from scipy.spatial.transform import Rotation

import pyvale.data as dataset
import pyvale.dataio as io
from pyvale import render

# %%
# 1. Load the deforming mesh and assign a speckle texture
# ------------------------------------------------------------

data_dir = dataset.riley_platehole_csv_case_path()

simulation = io.SimLoaderByField(
    load_dir=data_dir,
    coords_file="coords.csv",
    time_step_file=None,
    node_field_files={
        "disp_x": "field_disp_x.csv",
        "disp_y": "field_disp_y.csv",
        "disp_z": "field_disp_z.csv",
    },
    connect_files="connect.csv",
    load_opts=io.SimLoadOpts(
        coord_header=None,
        node_field_header=None,
    ),
).load_all_sim_data()

uvs = io.load_array(data_dir / "uvs.csv", header=None, delimiter=",")
texture = render.image_load(dataset.riley_speckle_texture_path())

mesh = render.mesh3d_from_simdata(
    simulation,
    shader=render.RileyTextureShader(uvs=uvs, texture=texture),
    displacement_keys=("disp_x", "disp_y", "disp_z"),
)

# We only render the first and last frame to save time. If you want to render 
# all frames comment this out.
frame_indices = render.first_last_frame_indices(mesh.displacements.shape[0])
mesh.displacements = render.select_frames(mesh.displacements, frame_indices)

# %%
# 2. Create and position a distorted stereo camera pair
# ------------------------------------------------------------
pixels_num = (2464, 2056)
pixels_size = (3.45e-6, 3.45e-6)
focal_length = 50.0e-3
roi_centre = tuple(render.mesh_center(mesh))

rot_world_0 = (0.0, 0.0, 0.0)
pos_world_0 = tuple(
    render.cam_pos_frame_points(
        mesh.coords,
        pixels_num,
        pixels_size,
        focal_length,
        rot_world_0,
        fov_scale=0.65,
    )
)

rot_world_1 = (0.0, float(np.deg2rad(20.0)), 0.0)
pos_world_1 = tuple(
    render.cam_pos_frame_points(
        mesh.coords,
        pixels_num,
        pixels_size,
        focal_length,
        rot_world_1,
        fov_scale=0.65,
    )
)

distortion_model = int(render.EDistortionModel.BROWN_CONRADY)

camera_0 = riley.Camera(
    pixels_num=pixels_num,
    pixels_size=pixels_size,
    pos_world=pos_world_0,
    rot_world=rot_world_0,
    roi_cent_world=roi_centre,
    focal_length=focal_length,
    sub_sample=2,
    distortion_model=distortion_model,
    distortion_k1=-0.2,
    distortion_k2=0.1,
    distortion_p1=0.0001,
    distortion_p2=-0.0001,
)

camera_1 = copy.deepcopy(camera_0)
camera_1.rot_world = rot_world_1
camera_1.pos_world = pos_world_1

# %%
# 3. Configure and build the renderer
# ------------------------------------------------------------

config = riley.create_raster_config(
    num_frames=mesh.displacements.shape[0],
    total_threads=8,
    save_strategy=riley.SaveStrategy.disk,
)
config.background_value = 128.0
config.tile_size_max = 128
config.save_scaling = riley.ScaleStrategy.none

output_dir = Path.cwd() / "pyvale-output" / "render3d_ex1d_riley_dicuq"

renderer = render.Riley(config, output_dir)

# %%
# 4. Build the scene and render the deforming specimen
# ------------------------------------------------------------
scene = render.Scene3D(meshes=[mesh], cameras=[camera_0, camera_1])
result = renderer.render(scene)

# %%
# 5. Save the stereo pair in Riley's exchange format
# ------------------------------------------------------------
# The calibration example loads its cameras from this file.
riley.save_stereo_pair(
    str(output_dir),
    "stereo_data_opengl.csv",
    camera_0,
    camera_1,
)

riley.save_stereo_pair(
    str(output_dir),
    "stereo_data_opencv.csv",
    replace(camera_0, coord_sys=riley.CameraCoordSys.opencv),
    replace(camera_1, coord_sys=riley.CameraCoordSys.opencv),
)
print(f"Rendered the stereo DIC images to {output_dir}")
print(f"{result.images=}")

# %%
# The first frame from both cameras is combined side by side below.
#
# .. image:: ../../../../_static/render3d_ex1d_riley_dicuq.png
#    :alt: Riley stereo DIC render from both cameras
#    :width: 900px
#    :align: center
