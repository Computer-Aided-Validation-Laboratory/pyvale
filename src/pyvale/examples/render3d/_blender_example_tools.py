# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Shared setup used by the Blender gallery examples."""

import numpy as np
from scipy.spatial.transform import Rotation

from pyvale.dataio import SimData
import pyvale.data as dataset
import pyvale.render as render


def build_scene() -> tuple[render.Mesh3D, render.Camera, list[render.Light]]:
    """Build the small textured Tri3 scene used in Blender examples."""
    # %% Intended render workflow: load or create SimData, then make a surface
    # Mesh3D. Volumes are skinned and conventions are enforced automatically.
    # Blender currently accepts Tri3 surface meshes only.
    sim_data = SimData(
        coords=np.array((
            (-20.0, -20.0, 0.0),
            (20.0, -20.0, 0.0),
            (-20.0, 20.0, 0.0),
        )),
        connect={"connect1": np.array(((0, 1, 2),))},
        node_vars={
            "disp_x": np.array(((0.0, 0.3), (0.0, 0.3), (0.0, 0.3))),
            "disp_y": np.zeros((3, 2)),
        },
    )

    camera = render.Camera(
        pixels_num=np.array((128, 128)),
        pixels_size=np.array((0.00345, 0.00345)),
        pos_world=np.array((0.0, 0.0, 500.0)),
        rot_world=Rotation.identity(),
        roi_cent_world=np.zeros(3),
        focal_length=15.0,
    )
    resolution = render.blender_mm_per_pixel(camera)
    mesh = render.mesh3d_from_simdata(
        sim_data,
        render.BlenderTextureShader(
            dataset.dic_pattern_5mpx_path(),
            resolution,
        ),
        ("disp_x", "disp_y"),
    )
    lights = [render.Light(
        render.ELightType.POINT, np.array((0.0, 0.0, 400.0)),
        np.zeros(3), 1.0,
    )]
    return mesh, camera, lights
