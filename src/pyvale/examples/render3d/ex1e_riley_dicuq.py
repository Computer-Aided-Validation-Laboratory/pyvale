"""Render a deforming stereo DIC experiment from CSV simulation data."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import riley

import pyvale.data as dataset
from pyvale import render

# %%
# 1. Load the deforming mesh and assign a speckle texture
# ------------------------------------------------------------
coords, connectivity, uvs, displacements = riley.load_sim_csvs(
    dataset.riley_platehole_csv_case_path()
)
displacements = displacements[[0, -1]]
texture = riley.load_texture_u8(dataset.riley_speckle_texture_path())
mesh = render.Mesh3D(
    element_type=render.EElementType.QUAD8,
    coords=coords,
    connectivity=connectivity,
    shader=render.RileyTextureShader(uvs=uvs, texture=texture),
    displacements=displacements,
)

# %%
# 2. Create and position a distorted stereo camera pair
# ------------------------------------------------------------
# The cameras are built natively so they can be saved in Riley's exchange
# format without any conversion of their parameters.
pixels_num = (2464, 2056)
pixels_size = (3.45e-6, 3.45e-6)
focal_length = 50.0e-3
roi_centre = riley.roi_cent_from_coords(coords)


def make_camera(angle_degrees: float) -> riley.Camera:
    """Create one distorted camera aimed at the specimen."""
    rot_world = (0.0, np.deg2rad(angle_degrees), 0.0)
    position = riley.pos_fill_frame_from_rot(
        coords,
        pixels_num,
        pixels_size,
        focal_length,
        rot_world,
        0.65,
    )
    return riley.Camera(
        pixels_num=pixels_num,
        pixels_size=pixels_size,
        pos_world=position,
        rot_world=rot_world,
        roi_cent_world=roi_centre,
        focal_length=focal_length,
        sub_sample=2,
        distortion_model=int(render.EDistortionModel.BROWN_CONRADY),
        distortion_k1=-0.2,
        distortion_k2=0.1,
        distortion_p1=0.0001,
        distortion_p2=-0.0001,
    )


camera_0 = make_camera(0.0)
camera_1 = make_camera(20.0)

# %%
# 3. Configure and build the renderer
# ------------------------------------------------------------
config = riley.create_raster_config(
    num_frames=displacements.shape[0],
    total_threads=8,
    save_strategy=riley.SaveStrategy.disk,
)
config.background_value = 128.0
config.tile_size_max = 128
config.save_scaling = riley.ScaleStrategy.none
output_dir = Path.cwd() / "pyvale-output" / "render-riley-dicuq"
renderer = render.Riley(config, output_dir)

# %%
# 4. Build the scene and render the deforming specimen
# ------------------------------------------------------------
scene = render.Scene3D(meshes=[mesh], cameras=[camera_0, camera_1])
result = renderer.render(scene)

# %%
# 5. Save the stereo pair in Riley's exchange format
# ------------------------------------------------------------
# The calibration example (ex1g) loads its cameras from this file.
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
