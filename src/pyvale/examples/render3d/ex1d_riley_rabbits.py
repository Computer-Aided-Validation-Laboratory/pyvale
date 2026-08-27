"""Render multiple mesh topologies and shader types with Riley."""

from pathlib import Path

import numpy as np
import riley
from riley.python import sceneops
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
mesh_groups: list[sceneops.MeshGroup] = []

for topology_index, (element_type, data_name) in enumerate(topologies):
    pair_start = len(meshes)
    for rabbit_name in ("riley", "feebs"):
        coords, connectivity, uvs = load_rabbit(rabbit_name, data_name)
        shader_index = len(meshes) % 3
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
                scaling=riley.ScaleStrategy.auto,
            )

        meshes.append(
            render.Mesh3D(
                element_type=element_type,
                coords=coords,
                connectivity=connectivity,
                shader=shader,
            )
        )

    sceneops.overlap_mesh_group_bounds(
        meshes,
        sceneops.mesh_group_single(pair_start),
        sceneops.mesh_group_single(pair_start + 1),
        sceneops.BoundsOverlapSpec(
            overlap_frac=(0.85, 0.8, 0.0),
            enabled_axes=(True, True, False),
            direct=(
                sceneops.EOverlapDirect.POSITIVE,
                sceneops.EOverlapDirect.NEGATIVE,
                sceneops.EOverlapDirect.CURRENT,
            ),
        ),
    )
    mesh_groups.append(sceneops.mesh_group_span(pair_start, 2))

sceneops.arrange_mesh_groups_grid(
    meshes,
    mesh_groups,
    sceneops.GridSpec(gap=(0.18, 0.28, 0.0), max_divs=(3, 2, 1)),
)

# %%
# 2. Create and position a camera around every mesh
# ------------------------------------------------------------
# Riley's camera helpers require native Riley meshes.
native_meshes = [render.to_native_mesh(mesh) for mesh in meshes]
pixels_num = np.array((1600, 800))
pixels_size = np.array((5.3e-6, 5.3e-6))
focal_length = 50.0e-3
camera = render.Camera(
    pixels_num=pixels_num,
    pixels_size=pixels_size,
    pos_world=np.asarray(
        riley.pos_fill_frame_from_rot_over_meshes(
            native_meshes,
            tuple(pixels_num),
            tuple(pixels_size),
            focal_length,
            (0.0, 0.0, 0.0),
            1.01,
        )
    ),
    rot_world=Rotation.identity(),
    roi_cent_world=np.asarray(riley.roi_cent_over_meshes(native_meshes)),
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
config.image_save_mode = riley.ImageSaveMode.grey
config.save_scaling = riley.ScaleStrategy.none
output_dir = Path.cwd() / "pyvale-output" / "render-riley-rabbits"
renderer = render.Riley(config, output_dir)

# %%
# 4. Build the multi-mesh scene and render it
# ------------------------------------------------------------
result = renderer.render(render.Scene3D(meshes=meshes, cameras=[camera]))
print(f"Rendered the rabbit topology comparison to {output_dir}")
print(f"{result.images=}")
