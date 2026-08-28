# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Riley: Speckle sphere
================================================================================
Render a sphere mesh of tri6 element with a speckle pattern texture shader.
"""

from pathlib import Path

import numpy as np
import riley
from scipy.spatial.transform import Rotation

import pyvale.data as dataset
import pyvale.dataio as io
from pyvale import render

# %%
# 1. Load the mesh and assign a texture shader
# ------------------------------------------------------------
data_dir: Path = dataset.riley_sphere200_case_path()
simulation = io.SimLoaderByField(
    load_dir=data_dir,
    coords_file="coords.csv",
    time_step_file=None,
    node_field_files=None,
    connect_files="connect.csv",
    load_opts=io.SimLoadOpts(coord_header=None),
).load_all_sim_data()

uvs = io.load_array(data_dir / "uvs.csv", header=None, delimiter=",")

texture = riley.load_texture_u8(dataset.riley_speckle_texture_path())

mesh = render.mesh3d_from_simdata(
    simulation,
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
    roi_cent_world=np.asarray(riley.roi_cent_from_coords(mesh.coords)),
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

render_engine = render.Riley(config, output_dir)

# %%
# 4. Build the scene and render it
# ------------------------------------------------------------
result = render_engine.render(render.Scene3D(meshes=[mesh], cameras=[camera]))

print(f"Rendered sphere render output to {output_dir}")
print(f"{result.images=}")
