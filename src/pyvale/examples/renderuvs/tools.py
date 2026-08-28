# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================
"""Shared rendering setup for documented Render UV examples."""

from pathlib import Path

import numpy as np
import riley
from scipy.spatial.transform import Rotation

from pyvale import render


def render_uv_example(
    mesh: render.Mesh3D,
    output_dir: Path,
    camera_rotation: Rotation,
) -> render.RenderResult:
    """Render one textured UV example with a camera framed around its mesh.

    The camera follows the supplied view rotation, looks at the mesh centre,
    and is positioned with :func:`render.cam_frame_mesh` to fill 95 percent
    of the frame.
    """
    mesh_center = render.mesh_center(mesh)
    camera = render.Camera(
        pixels_num=np.array((480, 320)),
        pixels_size=np.array((8.0e-6, 8.0e-6)),
        pos_world=np.zeros(3),
        rot_world=camera_rotation,
        roi_cent_world=mesh_center,
        focal_length=35.0e-3,
    )
    camera = render.cam_frame_mesh(camera, mesh, fill=0.95)
    camera = render.cam_look_at(
        camera,
        mesh_center,
        up=camera_rotation.apply(np.array((0.0, 1.0, 0.0))),
    )

    config = riley.create_raster_config(
        num_frames=1,
        total_threads=4,
        save_strategy=riley.SaveStrategy.disk,
    )
    config.background_value = 128.0
    renderer = render.Riley(config, output_dir)
    scene = render.Scene3D(meshes=[mesh], cameras=[camera])
    return renderer.render(scene)


__all__ = ["render_uv_example"]
