# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Riley: Feature Zoo
================================================================================

Exercise Riley's supported surface elements, shaders, image formats, camera
orientations, distortion, point-spread functions, and deformed frames through
pyvale's simulation-mesh conversion boundary.
"""

import shutil
from pathlib import Path

import numpy as np
import riley
from riley.pydemos import demo9_feature_zoo as zoo
from riley.python import sceneops

import pyvale.dataio as io
from pyvale import render

OUT_DIR = Path.cwd() / "pyvale-output" / "render3d_ex1h_riley_feature_zoo"


def build_meshes(channels: int, bits: int) -> list[render.Mesh3D]:
    """Build the feature-zoo meshes through the public pyvale API."""
    texture_name = (
        f"speck128_{'mono' if channels == 1 else 'rgb'}_u{bits}.png"
    )
    texture_path = riley.data.texture_dir_path() / texture_name
    if (channels, bits) == (1, 8):
        texture = riley.load_texture_mono_u8(texture_path)
    elif (channels, bits) == (1, 16):
        texture = riley.load_texture_mono_u16(texture_path)
    elif (channels, bits) == (3, 8):
        texture = riley.load_texture_rgb_u8(texture_path)
    else:
        texture_u8 = riley.load_texture_rgb_u8(
            riley.data.texture_dir_path() / "speck128_rgb_u8.png"
        )
        texture = texture_u8.astype(np.uint16) * np.uint16(257)

    meshes = []
    for index, (case_name, elem_type, mesh_type) in enumerate(zoo.CASES):
        coords, connect, uvs, temperature, disp_x, disp_y, disp_z = (
            zoo.load_case(case_name)
        )
        disp = (disp_x, disp_y, disp_z)
        shader = zoo.make_shader(
            index, channels, bits, uvs, temperature, disp, texture
        )
        simulation = io.SimData(
            coords=coords,
            connect={"surface": connect},
            node_vars={
                "disp_x": disp_x,
                "disp_y": disp_y,
                "disp_z": disp_z,
            },
        )
        convention = riley.ConnectConvention(
            elem_type,
            riley.EConnectAxis.ROW,
            0,
            riley.ENodeOrder.RILEY,
        )
        meshes.append(
            render.meshes3d_from_simdata(
                simulation,
                {"surface": convention},
                mesh_types={"surface": mesh_type},
                shaders={"surface": shader},
                displacement_keys=("disp_x", "disp_y", "disp_z")
                if index % 2
                else None,
            )["surface"]
        )

    plate = meshes[6]
    plate_x = np.array(plate.coords[:, 0], copy=True)
    plate.coords[:, 0] = -plate.coords[:, 1]
    plate.coords[:, 1] = plate_x
    mesh_coords = [mesh.coords for mesh in meshes]
    for index, center in enumerate(zoo.MESH_CENTERS):
        sceneops.scene_center_mesh_group_at(
            mesh_coords,
            sceneops.scene_create_mesh_group_single(index),
            center,
        )
    return meshes


def render_case(channels: int, bits: int) -> None:
    """Render one channel and bit-depth case."""
    meshes = build_meshes(channels, bits)
    native_meshes = [render.to_riley_mesh(mesh) for mesh in meshes]
    cameras = zoo.build_cameras(native_meshes)
    case_name = f"{'mono' if channels == 1 else 'rgb'}-u{bits}"
    config = riley.create_raster_config(
        num_frames=len(zoo.FRAME_INDICES),
        total_threads=4,
        save_strategy=riley.SaveStrategy.disk,
    )
    config.image_save_mode = (
        riley.ImageSaveMode.grey
        if channels == 1
        else riley.ImageSaveMode.rgb
    )
    config.save_bits = bits
    config.save_format = (
        riley.ImageFormat.bmp if bits == 8 else riley.ImageFormat.tiff
    )
    config.save_scaling = riley.ScaleStrategy.none
    config.background_value = 0.5 * (2**bits - 1)
    renderer = render.Riley(config, OUT_DIR / case_name)
    renderer.render(render.Scene3D(native_meshes, cameras))


shutil.rmtree(OUT_DIR, ignore_errors=True)
for channel_count, bit_count in ((1, 8), (1, 16), (3, 8), (3, 16)):
    render_case(channel_count, bit_count)

