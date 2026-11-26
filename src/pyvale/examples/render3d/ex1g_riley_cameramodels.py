# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Riley: Camera Models
====================

Compare how lens distortion and a point spread function affect the same
speckled sphere. All cameras share the same pose, resolution and pixel pitch.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import riley
from scipy.spatial.transform import Rotation

from pyvale import render

# %%
# Load a textured sphere and frame it with a perspective camera.

data_dir = riley.data.sphere200_case_path()
shader = riley.TextureShader(
    uvs=riley.load_csv(data_dir / "uvs.csv"),
    texture=riley.load_texture_mono_u8(riley.data.speckle_texture_path()),
)
mesh = render.Mesh3D(
    element_type=render.EElemType.TRI6,
    coords=riley.load_csv(data_dir / "coords.csv"),
    connectivity=riley.load_csv(data_dir / "connect.csv", dtype=np.int64),
    shader=riley.MeshShaderSet(mono=shader),
)
camera = render.Camera(
    pixels_num=np.array((800, 500)),
    pixels_size=np.array((5.3e-6, 5.3e-6)),
    pos_world=np.zeros(3),
    rot_world=Rotation.identity(),
    roi_cent_world=np.zeros(3),
    focal_length=50e-3,
    subsample=4,
)
camera = render.cam_frame_mesh(camera, mesh, fov_scale=1.0)

# %%
# The Brown–Conrady coefficients describe radial and tangential distortion.
# The anisotropic Gaussian models blur with different standard deviations
# along its principal axes, measured in pixels, and a rotation in radians.

distorted = replace(
    camera,
    distortion_model=render.EDistortionModel.BROWN_CONRADY,
    distortion_k1=-0.12,
    distortion_k2=0.035,
    distortion_p1=0.0002,
    distortion_p2=-0.0001,
)
blurred = replace(
    distorted,
    psf_type=render.EPSFType.ANISOTROPIC_GAUSSIAN,
    psf_sigma_x=1.2,
    psf_sigma_y=0.4,
    psf_theta=0.35,
    psf_support_rad=3.0,
    psf_separable=False,
)

config = riley.RasterConfig(parallel=4, save_strategy=riley.SaveStrategy.disk)
output = Path("pyvale-output") / "render3d_ex1g_riley_cameramodels"
scene = render.Scene3D([mesh], [camera, distorted, blurred])
render.Riley(config, output).render(scene)

# %%
# From left to right, the reference, distorted, and distorted and blurred images.
#
# .. image:: /_static/render3d_ex1g_riley_cameramodels.png
#    :alt: Speckled sphere with lens distortion and anisotropic blur
#    :width: 900px
#    :align: center
