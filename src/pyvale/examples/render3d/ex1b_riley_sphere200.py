"""Render a textured curved surface with Riley through ``pyvale.render``."""

from pathlib import Path

import numpy as np
import riley
from scipy.spatial.transform import Rotation

import pyvale.data as dataset
from pyvale import render

# %%
# 1. Load the mesh and assign a texture shader
# ------------------------------------------------------------
coords, connectivity, uvs, _ = riley.load_sim_csvs(
    dataset.riley_sphere200_case_path()
)
texture = riley.load_texture_u8(dataset.riley_speckle_texture_path())
mesh = render.Mesh3D(
    element_type=render.EElementType.TRI6,
    coords=coords,
    connectivity=connectivity,
    shader=render.RileyTextureShader(uvs=uvs, texture=texture),
)

# %%
# 2. Create and position the camera
# ------------------------------------------------------------
pixels_num = np.array((800, 500))
pixels_size = np.array((5.3e-6, 5.3e-6))
focal_length = 50.0e-3
rotation = Rotation.identity()
camera = render.Camera(
    pixels_num=pixels_num,
    pixels_size=pixels_size,
    pos_world=np.asarray(
        riley.pos_fill_frame_from_rot(
            coords,
            tuple(pixels_num),
            tuple(pixels_size),
            focal_length,
            tuple(rotation.as_euler("xyz")),
            1.0,
        )
    ),
    rot_world=rotation,
    roi_cent_world=np.asarray(riley.roi_cent_from_coords(coords)),
    focal_length=focal_length,
    subsample=2,
)

# %%
# 3. Configure and build the renderer
# ------------------------------------------------------------
config = riley.create_raster_config(
    num_frames=1,
    total_threads=4,
    save_strategy=riley.SaveStrategy.disk,
)
output_dir = Path.cwd() / "pyvale-output" / "render-riley-sphere200"
renderer = render.Riley(config, output_dir)

# %%
# 4. Build the scene and render it
# ------------------------------------------------------------
result = renderer.render(render.Scene3D(meshes=[mesh], cameras=[camera]))
print(f"Rendered sphere200 output to {output_dir}")
print(f"{result.images=}")
