# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Riley: Rendering quickstart
================================================================================
Compare Riley point-spread-function buffer modes through ``pyvale.render``.
"""

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
# 2. Create a camera with a Gaussian PSF
# ------------------------------------------------------------
pixels_num = np.array((800, 500))
pixels_size = np.array((5.3e-6, 5.3e-6))
focal_length = 50.0e-3
camera = render.Camera(
    pixels_num=pixels_num,
    pixels_size=pixels_size,
    pos_world=np.asarray(
        riley.pos_fill_frame_from_rot(
            coords,
            tuple(pixels_num),
            tuple(pixels_size),
            focal_length,
            (0.0, 0.0, 0.0),
            1.0,
        )
    ),
    rot_world=Rotation.identity(),
    roi_cent_world=np.asarray(riley.roi_cent_from_coords(coords)),
    focal_length=focal_length,
    subsample=2,
    psf_type=render.EPSFType.GAUSSIAN,
    psf_sigma_x=1.0,
    psf_support_rad=3.0,
)

# %%
# 3. Configure the two renderer variants
# ------------------------------------------------------------
output_root = Path.cwd() / "pyvale-output" / "render-riley-psf"
buffer_modes = (
    riley.BufferMode.global_subpx_full,
    riley.BufferMode.global_subpx_stripe,
)

# %%
# 4. Build and render the same scene with each configuration
# ------------------------------------------------------------
scene = render.Scene3D(meshes=[mesh], cameras=[camera])
for buffer_mode in buffer_modes:
    config = riley.create_raster_config(
        num_frames=1,
        total_threads=8,
        save_strategy=riley.SaveStrategy.disk,
    )
    config.buffer_mode = buffer_mode
    output_dir = output_root / buffer_mode.name
    result = render.Riley(config, output_dir).render(scene)
    print(f"Rendered {buffer_mode.name} output to {output_dir}")
    print(f"{result.images=}")
