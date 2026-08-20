# %%
"""Riley texture shader
======================

Use a backend-owned Riley texture shader with the common mesh and camera data.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.render as render
import riley


coords = np.array(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0),
                   (0.0, 1.0, 0.0)))
mesh = riley.Mesh(
    riley.MeshType.tri3, coords, np.array(((0, 1, 2),)),
    shader_type=riley.ShaderType.tex,
    uvs=np.array(((0.0, 0.0), (1.0, 0.0), (0.5, 1.0))),
    texture=np.array(((0, 255), (255, 0)), dtype=np.uint8),
)
camera = render.Camera(
    np.array((64, 64)), np.array((0.02, 0.02)), np.array((0.0, 0.0, 2.0)),
    Rotation.identity(), np.zeros(3), 1.0,
)
config = riley.create_raster_config(1, save_strategy=riley.SaveStrategy.both)
config.report = 0
output_dir = Path.cwd() / "pyvale-output" / "render-riley-texture"
result = render.Riley(config, output_dir).render(
    render.RenderScene((mesh,), (camera,)),
)
assert result.images is not None
print(result.images.shape)
