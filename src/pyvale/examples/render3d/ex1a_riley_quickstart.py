"""Riley quickstart
=================

Render a single triangle with Riley through the unified pyvale API.
"""


from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.render as render
import riley


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

camera = render.Camera(
    pixels_num=np.array((512, 512)),
    pixels_size=np.array((0.02, 0.02)),
    pos_world=np.array((0.0, 0.0, 2.0)),
    rot_world=Rotation.identity(),
    roi_cent_world=np.zeros(3),
    focal_length=1.0,
)

# Use Riley's camera auto-positioning to fill the FOV
pos_world = riley.pos_fill_frame_from_rot(
    coords,
    tuple(camera.pixels_num),
    tuple(camera.pixels_size),
    camera.focal_length,
    tuple(camera.rot_world.as_euler("xyz")),
    1.0,
)
camera.pos_world = np.array(pos_world)

config = riley.create_raster_config(
    1, save_strategy=riley.SaveStrategy.both
)
config.report = 1
config.background_value = 0.5

output_dir = Path.cwd() / "pyvale-output" / "render-riley-quickstart"
result = render.Riley(config, output_dir).render(
    render.RenderScene((mesh,), (camera,)),
)

assert result.images is not None
print(f"{result.images.shape=}")
