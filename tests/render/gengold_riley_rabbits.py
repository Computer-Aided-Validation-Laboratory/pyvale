# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Generate committed Riley rabbit multi-mesh image gold data."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.render as render
import riley


def build_rabbit_meshes() -> list[riley.Mesh]:
    """Load the two TRI3 rabbit meshes supplied with Riley."""
    meshes: list[riley.Mesh] = []
    texture = riley.load_texture(riley.data.speckle_texture_path())
    for rabbit_name in ("riley", "feebs"):
        data_path = riley.data.rabbit_case_path(rabbit_name, "tri3")
        coords = np.loadtxt(data_path / "coords.csv", delimiter=",")
        connectivity = np.loadtxt(
            data_path / "connectivity.csv", delimiter=",", dtype=np.uintp,
        )
        uvs = np.loadtxt(data_path / "uvs.csv", delimiter=",")
        meshes.append(riley.Mesh(
            riley.MeshType.tri3, coords, connectivity,
            shader_type=riley.ShaderType.tex, uvs=uvs, texture=texture,
        ))
    return meshes


def main() -> None:
    """Print a hash or write the trusted rabbit image array."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    meshes = build_rabbit_meshes()
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
    config = riley.create_raster_config(1, save_strategy=riley.SaveStrategy.memory)
    result = render.Riley(config).render(render.RenderScene(meshes, (camera,)))
    assert result.images is not None
    digest = hashlib.sha256(result.images.tobytes()).hexdigest()
    if arguments.write:
        path = Path(__file__).parent / "gold_riley/rabbits.npy"
        np.save(path, result.images)
        print(path)
    else:
        print(digest)


if __name__ == "__main__":
    main()
