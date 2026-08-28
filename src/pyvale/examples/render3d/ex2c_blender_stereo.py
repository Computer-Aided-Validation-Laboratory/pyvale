# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Blender Convergent Stereo Scene
================================================================================

This example demonstrates how to set up and render a convergent stereo camera
pair for 3D DIC using Blender and the unified pyvale render API.

Test case: mechanical analysis of a plate with a hole loaded in tension.

Workflow:
1. Load simulation data, scale units, and create a textured surface mesh.
2. Create base camera and generate convergent stereo cameras.
3. Add lighting to the scene.
4. Configure the Blender renderer backend.
5. Build Scene3D with both stereo cameras and render the stereo image pair.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.data as dataset
import pyvale.render as render
from pyvale.mooseherder import ExodusLoader
from pyvale.sensorsim import scale_length_units


# %%
# 1. Load simulation data and build a textured surface mesh
# --------------------------------------------------------------------------
data_path = dataset.render_mechanical_3d_path()
sim_data = ExodusLoader(data_path).load_all_sim_data()

disp_keys = ("disp_x", "disp_y", "disp_z")
sim_data = scale_length_units(1000.0, sim_data, disp_keys)

surface_mesh = render.mesh3d_from_simdata(
    sim_data,
    shader=None,
    displacement_keys=disp_keys,
)

# %%
# 2. Create base camera and generate convergent stereo cameras
# --------------------------------------------------------------------------
# Create the primary camera centered on the specimen ROI.
cam_base = render.Camera(
    pixels_num=np.array((1540, 1040)),
    pixels_size=np.array((0.00345, 0.00345)),
    pos_world=np.array((0.0, 0.0, 400.0)),
    rot_world=Rotation.identity(),
    roi_cent_world=np.zeros(3),
    focal_length=15.0,
)

resolution = render.blender_mm_per_pixel(cam_base)
surface_mesh.shader = render.BlenderTextureShader(
    image_path=dataset.dic_pattern_5mpx_path(),
    millimetres_per_pixel=resolution,
)

# Pyvale provides convenience helpers to build convergent stereo rigs:
# - symmetric_stereo_cameras: both cameras rotated symmetrically about ROI
# - faceon_stereo_cameras: cam0 normal to specimen, cam1 angled
stereo_angle = 15.0  # degrees
stereo = render.symmetric_stereo_cameras(cam_base, stereo_angle)

# %%
# 3. Create lighting
# --------------------------------------------------------------------------
light = render.Light(
    light_type=render.ELightType.POINT,
    pos_world=np.array((0.0, 0.0, 400.0)),
    direction_world=np.zeros(3),
    intensity=1.0,
)

# %%
# 4. Configure Blender renderer backend
# --------------------------------------------------------------------------
output_dir = Path.cwd() / "pyvale-output" / "render3d_ex2c_blender_stereo"
config = render.BlenderConfig(
    output_dir=output_dir,
    samples=4,
    threads=8,
    save_images=True,
    save_scene=True,
)
renderer = render.Blender(config)

# %%
# 5. Build Scene3D with both stereo cameras and render
# --------------------------------------------------------------------------
scene = render.Scene3D(
    meshes=[surface_mesh],
    cameras=[stereo.camera_0, stereo.camera_1],
    lights=[light],
)
result = renderer.render(scene)

print(f"Rendered {len(result.output_paths)} stereo images.")
print(f"Output saved to: {output_dir}")

# %%
# The first frame from both cameras is combined side by side below.
#
# .. image:: ../../../../_static/render3d_ex2c_blender_stereo.png
#    :alt: Blender stereo render from both cameras
#    :width: 900px
#    :align: center
