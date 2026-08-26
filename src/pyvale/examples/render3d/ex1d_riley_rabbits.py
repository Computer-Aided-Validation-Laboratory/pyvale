"""Render multiple mesh topologies and shader types with Riley."""

from pathlib import Path

import numpy as np
import riley
from scipy.spatial.transform import Rotation

import pyvale.data as dataset
from pyvale import render


def load_rabbit(
    rabbit: str,
    topology: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one static rabbit mesh and its UV coordinates."""
    data_dir = dataset.riley_rabbit_case_path(rabbit, topology)
    coords = np.loadtxt(data_dir / "coords.csv", delimiter=",")
    connectivity = np.loadtxt(
        data_dir / "connectivity.csv",
        delimiter=",",
        dtype=np.uintp,
    )
    uvs = np.loadtxt(data_dir / "uvs.csv", delimiter=",")
    return coords, connectivity, uvs


# %%
# 1. Load several mesh topologies and assign different shaders
# ------------------------------------------------------------
topologies = (
    (render.EElementType.TRI3, "tri3"),
    (render.EElementType.TRI6, "tri6"),
    (render.EElementType.QUAD4, "quad4"),
    (render.EElementType.QUAD8, "quad8"),
    (render.EElementType.QUAD9, "quad9"),
)
texture = riley.load_texture_u8(dataset.riley_speckle_texture_path())
meshes: list[render.Mesh3D] = []

for topology_index, (element_type, data_name) in enumerate(topologies):
    for rabbit_index, rabbit_name in enumerate(("riley", "feebs")):
        coords, connectivity, uvs = load_rabbit(rabbit_name, data_name)
        grid_index = 2 * topology_index + rabbit_index
        column = grid_index % 4
        row = grid_index // 4
        centre = 0.5 * (coords.min(axis=0) + coords.max(axis=0))
        coords = coords - centre + np.array((1.8 * column, -1.8 * row, 0.0))

        shader_index = grid_index % 3
        if shader_index == 0:
            shader = render.RileyTextureShader(uvs=uvs, texture=texture)
        elif shader_index == 1:
            nodal_field = 0.5 * (uvs[:, 0] + uvs[:, 1])
            shader = render.RileyNodalShader(
                field=nodal_field.reshape((1, -1, 1)),
            )
        else:
            shader = render.RileyFunctionShader(
                builtin=riley.FuncShaderBuiltin.checker,
                coord_mode=riley.FuncCoordMode.uv,
                parameters=riley.FuncShaderParams(coord_scale=(36.0, 36.0)),
                uvs=uvs,
            )

        meshes.append(
            render.Mesh3D(
                element_type=element_type,
                coords=coords,
                connectivity=connectivity,
                shader=shader,
            )
        )

# %%
# 2. Create and position a camera around every mesh
# ------------------------------------------------------------
all_coords = np.vstack([mesh.coords for mesh in meshes])
pixels_num = np.array((1600, 800))
pixels_size = np.array((5.3e-6, 5.3e-6))
focal_length = 50.0e-3
camera = render.Camera(
    pixels_num=pixels_num,
    pixels_size=pixels_size,
    pos_world=np.asarray(
        riley.pos_fill_frame_from_rot(
            all_coords,
            tuple(pixels_num),
            tuple(pixels_size),
            focal_length,
            (0.0, 0.0, 0.0),
            1.01,
        )
    ),
    rot_world=Rotation.identity(),
    roi_cent_world=np.asarray(riley.roi_cent_from_coords(all_coords)),
    focal_length=focal_length,
    subsample=2,
)

# %%
# 3. Configure and build the renderer
# ------------------------------------------------------------
config = riley.create_raster_config(
    num_frames=1,
    total_threads=1,
    save_strategy=riley.SaveStrategy.disk,
)
config.background_value = 127.5
config.save_scaling = riley.ScaleStrategy.none
output_dir = Path.cwd() / "pyvale-output" / "render-riley-rabbits"
renderer = render.Riley(config, output_dir)

# %%
# 4. Build the multi-mesh scene and render it
# ------------------------------------------------------------
result = renderer.render(render.Scene3D(meshes=meshes, cameras=[camera]))
print(f"Rendered the rabbit topology comparison to {output_dir}")
print(f"{result.images=}")
