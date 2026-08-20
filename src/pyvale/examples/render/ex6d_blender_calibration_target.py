# %%
"""Blender calibration-target images
=================================

Generate a compact stereo calibration-target sweep through the Blender backend.
Increase the angle and plunge limits to create a full calibration dataset.
"""

from pathlib import Path

import numpy as np

import pyvale.data as dataset
import pyvale.render as render
from _blender_example_tools import build_scene


_, camera, lights = build_scene()
stereo = render.CameraTools.faceon_stereo_cameras(camera, 15.0)
result = render.render_calibration_images(
    render.BlenderCalibrationTarget(
        np.array((15.0, 10.0, 1.0)), dataset.cal_target(), 0.1,
    ),
    stereo,
    render.BlenderConfig(
        Path.cwd() / "pyvale-output" / "render-blender-calibration-images",
    ),
    render.BlenderCalibrationData(
        angle_lims=(0.0, 0.0),
        angle_step=1.0,
        plunge_lims=(0.0, 0.0),
        plunge_step=1.0,
        x_limit=0.0,
        y_limit=0.0,
        max_images=10,
    ),
    lights,
)
print(f"Rendered {len(result.output_paths)} calibration images.")
