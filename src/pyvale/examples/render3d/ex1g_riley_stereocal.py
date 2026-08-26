"""Render a moving stereo-calibration target with Riley."""

from pathlib import Path

import numpy as np
import riley
from scipy.spatial.transform import Rotation

import pyvale.data as dataset
from pyvale import render

# %%
# 1. Load the moving calibration target and assign its texture
# ------------------------------------------------------------
coords, connectivity, uvs, displacements = riley.load_sim_csvs(
    dataset.riley_stereocal_case_path()
)
frame_indices = np.linspace(
    0,
    displacements.shape[0] - 1,
    min(8, displacements.shape[0]),
    dtype=int,
)
displacements = displacements[frame_indices]
texture = riley.load_texture_u8(dataset.riley_cal_target_texture_path())
mesh = render.Mesh3D(
    element_type=render.EElementType.TRI3,
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
    """Create one calibration camera."""
    rotation = Rotation.from_euler("y", angle_degrees, degrees=True)
    position = riley.pos_fill_frame_from_rot(
        coords,
        tuple(pixels_num),
        tuple(pixels_size),
        focal_length,
        tuple(rotation.as_euler("xyz")),
        0.8,
    )
    return render.Camera(
        pixels_num=pixels_num,
        pixels_size=pixels_size,
        pos_world=np.asarray(position),
        rot_world=rotation,
        roi_cent_world=roi_centre,
        focal_length=focal_length,
        subsample=2,
    )


cameras = [make_camera(0.0), make_camera(20.0)]

# %%
# 3. Configure and build the renderer
# ------------------------------------------------------------
config = riley.create_raster_config(
    num_frames=displacements.shape[0],
    total_threads=8,
    save_strategy=riley.SaveStrategy.disk,
)
config.background_value = 128.0
output_dir = Path.cwd() / "pyvale-output" / "render-riley-stereocal"
renderer = render.Riley(config, output_dir)

# %%
# 4. Build the scene and render every calibration-target pose
# ------------------------------------------------------------
result = renderer.render(render.Scene3D(meshes=[mesh], cameras=cameras))
print(f"Rendered stereo-calibration images to {output_dir}")
print(f"{result.images=}")
