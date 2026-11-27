# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Riley: Shader Pipeline Quickstart
=================================

Render a checkerboard, then use a shader pipeline to modulate it with a
spatially varying intensity. Both renders use the same mesh and camera.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import riley
from scipy.spatial.transform import Rotation

from pyvale import render

# %%
# Store the checkerboard in an output channel and a linear intensity
# gradient in a parameter register. The resolver multiplies them.
checker = riley.FunctionShader(
    builtin=riley.FuncShaderBuiltin.checker,
    coord_mode=riley.FuncCoordMode.world_reference,
    params=riley.FuncShaderParams(coord_scale=(4.0, 4.0)),
)
gradient = riley.FunctionShader(
    builtin=riley.FuncShaderBuiltin.linear,
    coord_mode=riley.FuncCoordMode.world_reference,
    params=riley.FuncShaderParams(linear_coeffs=(0.625, 0.375, 0.0)),
    bits=-1,
    scaling_type=riley.ScaleStrategy.none,
)
channel = riley.Register(riley.RegisterKind.channel, 0)
parameter = riley.Register(riley.RegisterKind.parameter, 0)
pipe = riley.ShaderPipe(
    stages=(
        riley.ShaderSource(checker, channel),
        riley.ShaderSource(gradient, parameter),
    ),
    resolver=riley.ShaderResolver(
        riley.ResolverKind.modulate, registers=(channel,), factor=parameter,
    ),
)

# %%
# Render both versions with fixed file scaling so the change in
# intensity is visible in the saved images.
mesh = render.Mesh3D(
    element_type=render.EElemType.TRI3,
    coords=np.array(((-1., -1., 0.), (1., -1., 0.), (0., 1., 0.))),
    connectivity=np.array(((0, 1, 2),)),
    shader=riley.MeshShaderSet(mono=checker),
)
camera = render.Camera(
    pixels_num=np.array((512, 512)),
    pixels_size=np.array((0.02, 0.02)),
    pos_world=np.array((0., 0., 2.)),
    rot_world=Rotation.identity(),
    roi_cent_world=np.zeros(3),
    focal_length=1.0,
    subsample=4,
    output_files=(riley.ImageSaveOpts(
        scaling=riley.ScaleStrategy.fixed, scaling_min=0.0, scaling_max=255.0,
    ),),
)
camera = render.cam_frame_mesh(camera, mesh, fov_scale=1.0)
config = riley.RasterConfig(parallel=1, save_strategy=riley.SaveStrategy.disk)
output = Path("pyvale-output") / "render3d_ex1c_riley_shaderpipe_quickstart"
modulated = replace(mesh, shader=riley.MeshShaderSet(mono=pipe))
render.Riley(config, output / "direct").render(render.Scene3D([mesh], [camera]))
render.Riley(config, output / "modulated").render(
    render.Scene3D([modulated], [camera]),
)

# %%
# .. image:: /_static/render3d_ex1c_riley_shaderpipe_quickstart.png
#    :alt: Direct checkerboard and checkerboard modulated by an intensity gradient
#    :width: 750px
#    :align: center
