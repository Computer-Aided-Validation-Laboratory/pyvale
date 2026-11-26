# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Riley: Procedural Speckle Quickstart
====================================

Generate a repeatable disk speckle pattern on a plane. Speckle radius
is set in the plane's physical units, independently of image resolution.
"""

from pathlib import Path

import numpy as np
import riley
from scipy.spatial.transform import Rotation

from pyvale import render

# %%
# Assign each corner of a 40-unit square its UV coordinate.
coords = np.array(((-20., -20., 0.), (20., -20., 0.),
                   (20., 20., 0.), (-20., 20., 0.)))
uvs = np.array(((0., 1.), (1., 1.), (1., 0.), (0., 0.)))
speckle = riley.SpeckleShader(
    uvs=uvs,
    params=riley.Disk2DParams(coverage=0.5, radius=0.25),
)
mesh = render.Mesh3D(
    element_type=render.EElemType.QUAD4,
    coords=coords,
    connectivity=np.array(((0, 1, 2, 3),)),
    shader=riley.MeshShaderSet(mono=speckle),
)

# %%
# The front-facing camera frames the square. Its file settings convert
# the shader's intensity range of 0 to 1 to 8-bit grayscale.
pixels = np.array((1024, 1024))
focal_length = 50e-3
camera = render.Camera(
    pixels_num=pixels,
    pixels_size=focal_length / pixels,
    pos_world=np.array((0., 0., 40.)),
    rot_world=Rotation.identity(),
    roi_cent_world=np.zeros(3),
    focal_length=focal_length,
    output_files=(riley.ImageSaveOpts(
        format=riley.ImageFormat.bmp,
        file_bits=8,
        scaling=riley.ScaleStrategy.fixed,
        scaling_min=0.0,
        scaling_max=1.0,
    ),),
)
config = riley.RasterConfig(parallel=4, save_strategy=riley.SaveStrategy.disk)
output = Path("pyvale-output") / "render3d_ex1b_riley_procedural_quickstart"
render.Riley(config, output).render(render.Scene3D([mesh], [camera]))

# %%
# .. image:: /_static/render3d_ex1b_riley_procedural_quickstart.png
#    :alt: Disk speckle rendered on a square plane
#    :width: 750px
#    :align: center
