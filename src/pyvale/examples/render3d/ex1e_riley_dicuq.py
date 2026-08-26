"""Render a deforming stereo DIC experiment from CSV simulation data."""

from pathlib import Path

import numpy as np
import riley
from scipy.spatial.transform import Rotation

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
# 2. Create and position a stereo camera pair
# ------------------------------------------------------------
pixels_num = np.array((2464, 2056))
pixels_size = np.array((3.45e-6, 3.45e-6))
focal_length = 50.0e-3
roi_centre = np.asarray(riley.roi_cent_from_coords(coords))


def make_camera(angle_degrees: float) -> render.Camera:
    """Create one distorted camera aimed at the specimen."""
    rotation = Rotation.from_euler("y", angle_degrees, degrees=True)
    position = riley.pos_fill_frame_from_rot(
        coords,
        tuple(pixels_num),
        tuple(pixels_size),
        focal_length,
        tuple(rotation.as_euler("xyz")),
        0.65,
    )
    return render.Camera(
        pixels_num=pixels_num,
        pixels_size=pixels_size,
        pos_world=np.asarray(position),
        rot_world=rotation,
        roi_cent_world=roi_centre,
        focal_length=focal_length,
        subsample=2,
        distortion_model=render.EDistortionModel.BROWN_CONRADY,
        distortion_k1=-0.2,
        distortion_k2=0.1,
        distortion_p1=0.0001,
        distortion_p2=-0.0001,
    )


camera_0 = make_camera(0.0)
camera_1 = make_camera(20.0)
stereo = render.CameraStereo(camera_0, camera_1)

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
# 4. Build the scene, render it, and save the stereo calibration
# ------------------------------------------------------------
scene = render.Scene3D(meshes=[mesh], cameras=[camera_0, camera_1])
result = renderer.render(scene)
stereo.save_calibration(output_dir)
print(f"Rendered the stereo DIC images to {output_dir}")
print(f"{result.images=}")
