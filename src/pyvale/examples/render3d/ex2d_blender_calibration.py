# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================

"""
Blender: Stereo Camera Calibration & Parameters
================================================================================

This tutorial demonstrates how to define a stereo camera system, export its
intrinsic and extrinsic calibration parameters in standard formats (YAML and
MatchID), and reload calibration data.

Stereo calibration defines:
- Intrinsic parameters: focal length, pixel size, sensor grid, principal point.
- Extrinsic parameters: 3D rotation and translation vector between cameras.

Workflow:
1. Define a base perspective camera and construct a convergent stereo system.
2. Export the ground-truth stereo calibration to YAML and MatchID formats.
3. Reload calibration data to verify parameter fidelity.
4. Calculate calibration image counts for physical target pose sweeps.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.render as render


# %%
# 1. Define base camera and build convergent stereo system
# --------------------------------------------------------------------------
cam_base = render.Camera(
    pixels_num=np.array((1540, 1040)),
    pixels_size=np.array((0.00345, 0.00345)),
    pos_world=np.array((0.0, 0.0, 400.0)),
    rot_world=Rotation.identity(),
    roi_cent_world=np.zeros(3),
    focal_length=15.0,
)

stereo_angle = 15.0  # degrees
stereo = render.faceon_stereo_cameras(cam_base, stereo_angle)

print("Stereo baseline distance (mm):", stereo.stereo_dist)
print("Stereo rotation (Euler xyz deg):")
print(stereo.stereo_rotation.as_euler("xyz", degrees=True))

# %%
# 2. Export calibration in YAML and MatchID formats
# --------------------------------------------------------------------------
output_dir = (
    Path.cwd() / "pyvale-output"
    / "render3d_ex2d_blender_calibration"
)
output_dir.mkdir(parents=True, exist_ok=True)

# Save standard Pyvale YAML calibration:
stereo.save_calibration(output_dir)
yaml_path = output_dir / "calibration" / "calibration.yaml"
print(f"Saved Pyvale calibration to: {yaml_path}")

# Save MatchID compatible calibration:
stereo.save_calibration_mid(output_dir)
mid_path = output_dir / "calibration" / "calibration.caldat"
print(f"Saved MatchID calibration to: {mid_path}")

# %%
# 3. Reload and inspect calibration parameters
# --------------------------------------------------------------------------
reloaded = render.CameraStereo.from_calibration(
    calib_path=yaml_path,
    pos_world_0=cam_base.pos_world,
    rot_world_0=cam_base.rot_world,
    focal_length=cam_base.focal_length,
)
assert np.allclose(reloaded.stereo_dist, stereo.stereo_dist)
print("Successfully verified round-trip calibration parameters.")

# %%
# 4. Plan calibration target pose sweeps
# --------------------------------------------------------------------------
# For experimental DIC calibration, targets are translated and rotated
# through the field of view. We can compute the total pose count:
cal_data = render.BlenderCalibrationData(
    angle_lims=(-10.0, 10.0),
    angle_step=5.0,
    plunge_lims=(-5.0, 5.0),
    plunge_step=5.0,
)
total_images = render.calibration_image_count(cal_data)
print(f"Standard calibration sweep image count: {total_images}")

# %%
# The generated Pyvale calibration file is embedded below.
#
# .. literalinclude:: ../../../../_static/render3d_ex2d_blender_calibration.yaml
#    :language: yaml
#    :caption: Generated stereo camera calibration
