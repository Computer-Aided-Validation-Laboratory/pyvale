# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Shared setup used by the Blender gallery examples."""

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.dataset as dataset
import pyvale.mooseherder as mooseherder
import pyvale.render as render
import pyvale.sensorsim as sensorsim


def build_scene() -> tuple[render.Mesh, render.Camera, list[render.Light]]:
    """Build the small textured plate scene used in Blender examples."""
    sim_data = mooseherder.ExodusLoader(
        dataset.mechanical_2d_path(),
    ).load_all_sim_data()
    sensorsim.scale_length_units(1000.0, sim_data, ("disp_x", "disp_y"))

    camera = render.Camera(
        np.array((128, 128)), np.array((0.00345, 0.00345)),
        np.array((0.0, 0.0, 500.0)), Rotation.identity(), np.zeros(3), 15.0,
    )
    resolution = render.CameraTools.blender_mm_per_pixel(camera)
    mesh = render.blender.mesh_from_simdata(
        sim_data,
        render.BlenderTextureShader(dataset.dic_pattern_5mpx_path(), resolution),
        ("disp_x", "disp_y"),
    )
    lights = [render.Light(
        render.ELightType.POINT, np.array((0.0, 0.0, 400.0)),
        np.zeros(3), 1.0,
    )]
    return mesh, camera, lights
