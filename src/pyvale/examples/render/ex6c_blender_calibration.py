# %%
"""Blender stereo calibration
==========================

Create and save the YAML calibration associated with a Blender stereo setup.
"""

from pathlib import Path

import pyvale.render as render
from _blender_example_tools import build_scene


output_dir = Path("pyvale-output/blender-calibration")
output_dir.mkdir(parents=True, exist_ok=True)
_, camera, _ = build_scene()
stereo = render.CameraTools.faceon_stereo_cameras(camera, 15.0)
stereo.save_calibration(output_dir)
print(render.calibration_image_count(render.BlenderCalibrationData()))
