# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Rendering quickstart with Riley
================================================================================

Here we render a single triangle with Riley through the unified pyvale render
module.

The common workflow for a rendering simulation is pyvale is:
1. Load and/or create meshes assigning shaders and displacement fields
2. Create cameras and position them in the scene
3. Set render configuration and use this to build the render backend
4. Build the scene then render images to memory and/or disk

Riley is the default rendering backend for digital image correlation (DIC)
uncertainty quantification with pyvale. Riley supports full 3D rendering for
stereo DIC but is also the default for 2D image simulation due to its
computation speed and verification suite independent of DIC.
"""

from pathlib import Path

import numpy as np
import riley
from scipy.spatial.transform import Rotation

from pyvale import render

# %%
# 1. Make the triangle mesh
# -------------------------------

coords = np.array(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)))
connect = np.array(((0, 1, 2),))

mesh = render.Mesh3D(
    element_type=render.EElementType.TRI3,
    coords=coords,
    connectivity=connect,
    shader=render.RileyFunctionShader(
        builtin=riley.FuncShaderBuiltin.checker,
        coord_mode=riley.FuncCoordMode.world_reference,
    ),
)

# %%
# 2. Create and position a camera
# -------------------------------

camera = render.Camera(
    pixels_num=np.array((512, 512)),
    pixels_size=np.array((0.02, 0.02)),
    pos_world=np.array((0.0, 0.0, 2.0)),
    rot_world=Rotation.identity(),
    roi_cent_world=np.zeros(3),
    focal_length=1.0,
)

pos_world = riley.pos_fill_frame_from_rot(
    coords,
    tuple(camera.pixels_num),
    tuple(camera.pixels_size),
    camera.focal_length,
    tuple(camera.rot_world.as_euler("xyz")),
    1.0,
)
camera.pos_world = np.array(pos_world)

# %%
# 3. Build renderer backend
# -------------------------------

config = riley.create_raster_config(1, save_strategy=riley.SaveStrategy.both)
config.report = 1
config.background_value = 0.5
output_dir = Path.cwd() / "pyvale-output" / "render-riley-quickstart"
renderer = render.Riley(config, output_dir)

# %%
# 4. Render images from scene
# -------------------------------

scene = render.Scene3D([mesh], [camera])
result = renderer.render(scene)

assert result.images is not None
print(f"{result.images.shape=}")
