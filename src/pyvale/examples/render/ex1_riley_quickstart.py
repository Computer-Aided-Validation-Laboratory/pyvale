# %%
"""Riley quickstart
=================

Render a single triangle with Riley through the unified pyvale API.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path

import pyvale.render as render
import riley


coords = np.array(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0),
                   (0.0, 1.0, 0.0)))
mesh = riley.Mesh(
    riley.MeshType.tri3,
    coords,
    np.array(((0, 1, 2),)),
    shader_type=riley.ShaderType.func,
    func_shader_builtin=riley.FuncShaderBuiltin.checker,
    func_shader_coord_mode=riley.FuncCoordMode.world_reference,
)
camera = render.Camera(
    pixels_num=np.array((64, 64)),
    pixels_size=np.array((0.02, 0.02)),
    pos_world=np.array((0.0, 0.0, 2.0)),
    rot_world=Rotation.identity(),
    roi_cent_world=np.zeros(3),
    focal_length=1.0,
)
config = riley.create_raster_config(1, save_strategy=riley.SaveStrategy.memory)
output_dir = Path.cwd() / "pyvale-output" / "render-riley-quickstart"
result = render.Riley(config, output_dir).render(
    render.RenderScene((mesh,), (camera,)),
)
assert result.images is not None
print(result.images.shape)
