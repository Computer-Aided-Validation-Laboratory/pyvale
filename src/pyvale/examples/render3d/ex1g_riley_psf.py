# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Riley: Point spread functions
================================================================================
Compare Riley point spread function buffer modes through ``pyvale.render``.
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
data_dir = dataset.riley_sphere200_case_path()
simulation = io.SimLoaderByField(
    load_dir=data_dir,
    coords_file="coords.csv",
    time_step_file=None,
    node_field_files=None,
    connect_files="connect.csv",
    load_opts=io.SimLoadOpts(coord_header=None),
).load_all_sim_data()

uvs = io.load_array(data_dir / "uvs.csv", header=None, delimiter=",")
texture = render.image_load(dataset.riley_speckle_texture_path())

shader = render.RileyTextureShader(uvs=uvs, texture=texture)
mesh = render.mesh3d_from_simdata(
    simulation,
    shader=shader,
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
    pos_world=np.zeros(3),
    rot_world=Rotation.identity(),
    roi_cent_world=np.zeros(3),
    focal_length=focal_length,
    subsample=2,
    psf_type=render.EPSFType.GAUSSIAN,
    psf_sigma_x=1.0,
    psf_support_rad=3.0,
)
camera = render.cam_frame_mesh(camera, mesh, fill=1.0)

# %%
# 3. Configure the two renderer variants
# ------------------------------------------------------------
output_root = Path.cwd() / "pyvale-output" / "render3d_ex1g_riley_psf"

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
        total_threads=4,
        save_strategy=riley.SaveStrategy.disk,
    )
    
    config.buffer_mode = buffer_mode
    output_dir = output_root / buffer_mode.name
    renderer = render.Riley(config, output_dir)
    result = renderer.render(scene)

    print(f"Rendered {buffer_mode.name} output to {output_dir}")
    print(f"{result.images=}")

# %%
# The ``global_subpx_full`` result is used as the representative buffer mode.
#
# .. image:: ../../../../_static/render3d_ex1g_riley_psf.png
#    :alt: Speckled sphere rendered with a Gaussian point spread function
#    :width: 500px
#    :align: center
