"""Riley: Shader pipelines and channel formats
================================================================================

A shader pipeline combines sources before a camera selects mono, RGB or
multifield output. Values here remain in floating point without automatic
scaling; file output is configured separately on each camera.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import riley
from scipy.spatial.transform import Rotation

from pyvale import render

source = riley.FunctionShader(
    builtin=riley.FuncShaderBuiltin.constant,
    params=riley.FuncShaderParams(constant_value=0.25),
    bits=-1, scaling_type=riley.ScaleStrategy.none,
)
factor = riley.FunctionShader(
    builtin=riley.FuncShaderBuiltin.constant,
    params=riley.FuncShaderParams(constant_value=0.5),
    bits=-1, scaling_type=riley.ScaleStrategy.none,
)
channel = riley.Register(riley.RegisterKind.channel, 0)
parameter = riley.Register(riley.RegisterKind.parameter, 0)
pipe = riley.ShaderPipe(
    stages=(
        riley.ShaderSource(source, channel),
        riley.ShaderSource(factor, parameter),
    ),
    resolver=riley.ShaderResolver(
        riley.ResolverKind.modulate, registers=(channel,), factor=parameter,
    ),
)
field = np.empty((3, 1, 3))
field[:] = (0.2, 0.5, 0.8)
mesh = render.Mesh3D(
    element_type=render.EElemType.TRI3,
    coords=np.array(((-1., -1., 0.), (1., -1., 0.), (0., 1., 0.))),
    connectivity=np.array(((0, 1, 2),)),
    shader=riley.MeshShaderSet(
        mono=pipe,
        rgb=riley.NodalShader(field, bits=-1, scaling_type=riley.ScaleStrategy.none),
        multifield=riley.NodalShader(field, bits=-1, scaling_type=riley.ScaleStrategy.none),
    ),
)
camera = render.Camera(
    pixels_num=np.array((96, 64)), pixels_size=np.array((0.02, 0.02)),
    pos_world=np.array((0., 0., 2.)), rot_world=Rotation.identity(),
    roi_cent_world=np.zeros(3), focal_length=1.,
    output_files=(riley.ImageSaveOpts(scaling=riley.ScaleStrategy.auto),),
)
config = riley.RasterConfig(parallel=1, save_strategy=riley.SaveStrategy.both)
output = Path.cwd() / 'pyvale-output' / 'render3d_ex1i_riley_shader_pipeline'
for name, format_, channels in (
    ('mono', riley.RenderFormat.mono, 1),
    ('rgb', riley.RenderFormat.rgb, 3),
    ('multifield', riley.RenderFormat.multifield, 3),
):
    selected = replace(camera, render_format=format_, render_channels=channels)
    result = render.Riley(config, output / name).render(render.Scene3D([mesh], [selected]))
    assert result.images.shape == (1, 1, 64, 96, channels)
    expected = (0.125,) if channels == 1 else (0.2, 0.5, 0.8)
    np.testing.assert_allclose(result.images[0, 0, 32, 48], expected)
    print(name, result.images.shape)
