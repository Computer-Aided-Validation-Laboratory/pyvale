# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================


"""Riley: Multi-Shader, Multi-Element Rabbits
================================================================================

Here we render a series of rabbit meshes of all different element types cycling
through all support shaders for Riley to show how to build a scene with mixed
element and shader types.
"""

from pathlib import Path

import numpy as np
import riley
from riley.python import sceneops
from scipy.spatial.transform import Rotation

import pyvale.data as dataset
import pyvale.dataio as io
from pyvale import render


def load_rabbit(
    rabbit: str,
    topology: render.EElementType,
) -> tuple[io.SimData, np.ndarray]:
    """Load one static rabbit mesh and its UV coordinates."""
    data_dir = dataset.riley_rabbit_case_path(rabbit, topology)

    simulation = io.SimLoaderByField(
        load_dir=data_dir,
        coords_file="coords.csv",
        time_step_file=None,
        node_field_files=None,
        connect_files="connectivity.csv",
        load_opts=io.SimLoadOpts(coord_header=None),
    ).load_all_sim_data()

    uvs = io.load_array(data_dir / "uvs.csv", header=None, delimiter=",")

    return simulation, uvs


# %%
# 1. Load several mesh topologies and assign different shaders
# ------------------------------------------------------------

topologies = (
    render.EElementType.TRI3,
    render.EElementType.TRI6,
    render.EElementType.QUAD4,
    render.EElementType.QUAD8,
    render.EElementType.QUAD9,
)

texture = riley.load_texture_u8(dataset.riley_speckle_texture_path())

meshes: list[render.Mesh3D] = []
mesh_groups: list[sceneops.MeshGroup] = []

for topology_index, element_type in enumerate(topologies):
    pair_start = len(meshes)
    for rabbit_name in ("riley", "feebs"):
        simulation, uvs = load_rabbit(rabbit_name, element_type)
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

        mesh = render.mesh3d_from_simdata(simulation, shader=shader)

        if mesh.element_type is not element_type:
            raise ValueError(
                f"Unexpected topology loaded for {element_type}."
            )

        meshes.append(mesh)

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

pixels_num = np.array((1600, 800))
pixels_size = np.array((5.3e-6, 5.3e-6))
focal_length = 50.0e-3

camera = render.Camera(
    pixels_num=pixels_num,
    pixels_size=pixels_size,
    pos_world=np.zeros(3),
    rot_world=Rotation.identity(),
    roi_cent_world=np.zeros(3),
    focal_length=focal_length,
    subsample=2,
)
camera = render.cam_frame_scene(camera, meshes, fill=1.01)

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

output_dir = Path.cwd() / "pyvale-output" / "render3d_ex1c_riley_rabbits"

renderer = render.Riley(config, output_dir)

# %%
# 4. Build the multi-mesh scene and render it
# ------------------------------------------------------------

result = renderer.render(render.Scene3D(meshes=meshes, cameras=[camera]))

print(f"Rendered the rabbit topology comparison to {output_dir}")
print(f"{result.images=}")

# %%
# The first rendered frame containing all topology variants is shown below.
#
# .. image:: ../../../../_static/render3d_ex1c_riley_rabbits.png
#    :alt: Riley rabbit meshes rendered with several element topologies
#    :width: 700px
#    :align: center
