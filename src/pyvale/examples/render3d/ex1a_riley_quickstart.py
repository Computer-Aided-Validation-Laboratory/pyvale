# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Renderin quickstart with Riley
================================================================================

Render a single triangle with Riley through the unified pyvale render module.
"""


from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.render as render
import riley

#%%
# 1. Make the triangle mesh
# -------------------------------

coords = np.array(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0),
                   (0.0, 1.0, 0.0)))
connect = np.array(((0, 1, 2),))

mesh = riley.Mesh(
    riley.MeshType.tri3,
    coords,
    connect,
    shader_type=riley.ShaderType.func,
    func_shader_builtin=riley.FuncShaderBuiltin.checker,
    func_shader_coord_mode=riley.FuncCoordMode.world_reference,
)

#%%
# 2. Build and position a camera
# -------------------------------
# Use Riley's camera auto-positioning to fill the FOV

camera = render.Camera(
    pixels_count=np.array((512, 512)),
    pixel_size=np.array((0.02, 0.02)),
    pos_world=np.array((0.0, 0.0, 2.0)),
    rot_world=Rotation.identity(),
    roi_cent_world=np.zeros(3),
    focal_length=1.0,
)

pos_world = riley.pos_fill_frame_from_rot(
    coords,
    tuple(camera.pixels_count),
    tuple(camera.pixel_size),
    camera.focal_length,
    tuple(camera.rot_world.as_euler("xyz")),
    1.0,
)
camera.pos_world = np.array(pos_world)

#%%
# 1. Load physics simulation data
# -------------------------------

config = riley.create_raster_config(
    1, save_strategy=riley.SaveStrategy.both
)
config.report = 1
config.background_value = 0.5
output_dir = Path.cwd() / "pyvale-output" / "render-riley-quickstart"


#%%
# 1. Load physics simulation data
# -------------------------------

result = render.Riley(config, output_dir).render(
    render.Scene3D([mesh], [camera]),
)

assert result.images is not None
print(f"{result.images.shape=}")
