# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ============================================================================

"""Private scene construction for the Render UV gallery examples."""

from pathlib import Path

import numpy as np
import riley
from scipy.spatial.transform import Rotation

from pyvale import render
from pyvale import verif


TEXTURE_SHAPE = (2056, 2464)


def rectangle_grid(
    length_u: float = 2.0,
    length_v: float = 1.0,
    elements_u: int = 12,
    elements_v: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a centred Quad4 verification grid in the XY plane."""
    coords, connectivity = verif.rectangle_mesh_2d(
        length_u, length_v, elements_u, elements_v,
    )
    coords[:, 0] -= 0.5 * length_u
    coords[:, 1] -= 0.5 * length_v
    return coords, connectivity


def embed_grid(
    coords: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    origin: np.ndarray | None = None,
) -> np.ndarray:
    """Embed XY grid coordinates into a three-dimensional plane."""
    if origin is None:
        origin = np.zeros(3)
    return (
        origin
        + coords[:, [0]] * np.asarray(axis_u)
        + coords[:, [1]] * np.asarray(axis_v)
    )


def render_uv_variant(
    display_coords: np.ndarray,
    connectivity: np.ndarray,
    uvs: np.ndarray,
    texture: np.ndarray,
    output_dir: Path,
) -> None:
    """Render one static textured mesh to a named example subdirectory."""
    element_types = {
        3: render.EElementType.TRI3,
        4: render.EElementType.QUAD4,
        6: render.EElementType.TRI6,
        8: render.EElementType.QUAD8,
        9: render.EElementType.QUAD9,
    }
    nodes_per_element = connectivity.shape[1]
    if nodes_per_element not in element_types:
        raise ValueError(
            f"Unsupported connectivity with {nodes_per_element} nodes.",
        )
    shader = render.RileyTextureShader(uvs=uvs, texture=texture)
    mesh = render.Mesh3D(
        element_type=element_types[nodes_per_element],
        coords=display_coords,
        connectivity=connectivity,
        shader=shader,
    )

    pixels_num = np.array((480, 320))
    pixels_size = np.array((8.0e-6, 8.0e-6))
    focal_length = 35.0e-3
    camera = render.Camera(
        pixels_num=pixels_num,
        pixels_size=pixels_size,
        pos_world=np.zeros(3),
        rot_world=Rotation.identity(),
        roi_cent_world=np.zeros(3),
        focal_length=focal_length,
    )
    camera = render.cam_frame_mesh(camera, mesh, fill=0.9)

    config = riley.create_raster_config(
        num_frames=1,
        total_threads=4,
        save_strategy=riley.SaveStrategy.disk,
    )
    config.background_value = 128.0
    scene = render.Scene3D(meshes=[mesh], cameras=[camera])
    renderer = render.Riley(config, output_dir)
    renderer.render(scene)


__all__ = [
    "TEXTURE_SHAPE", "embed_grid", "rectangle_grid", "render_uv_variant",
]
