# %%
"""Riley rabbit multi-mesh scene
===============================

Render two Riley rabbit meshes through pyvale's common mesh and camera API.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.render as render
import riley


texture = riley.load_texture(riley.data.speckle_texture_path())
meshes = []
for rabbit_name in ("riley", "feebs"):
    data_path = riley.data.rabbit_case_path(rabbit_name, "tri3")
    meshes.append(riley.Mesh(
        riley.MeshType.tri3,
        np.loadtxt(data_path / "coords.csv", delimiter=","),
        np.loadtxt(data_path / "connectivity.csv", delimiter=",", dtype=np.uintp),
        shader_type=riley.ShaderType.tex,
        uvs=np.loadtxt(data_path / "uvs.csv", delimiter=","),
        texture=texture,
    ))

coords = np.concatenate([mesh.coords for mesh in meshes])
pixels_num = np.array((320, 160))
pixels_size = np.array((5.3e-6, 5.3e-6))
focal_length = 50.0e-3
rotation = Rotation.from_euler("xyz", (0.0, np.pi, 0.0))
position = riley.pos_fill_frame_from_rot(
    coords, tuple(pixels_num), tuple(pixels_size), focal_length,
    tuple(rotation.as_euler("xyz")), 1.1,
)
camera = render.Camera(
    pixels_num, pixels_size, np.asarray(position), rotation,
    np.mean(coords, axis=0), focal_length,
)
config = riley.create_raster_config(1, save_strategy=riley.SaveStrategy.both)
config.report = 0
output_dir = Path.cwd() / "pyvale-output" / "render-riley-rabbits"
result = render.Riley(config, output_dir).render(
    render.RenderScene(meshes, (camera,)),
)
assert result.images is not None
print(result.images.shape)
