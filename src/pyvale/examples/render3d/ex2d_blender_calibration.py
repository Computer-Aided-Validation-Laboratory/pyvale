"""Blender stereo calibration
==========================

Create and save the YAML calibration associated with a Blender stereo setup.
"""


from pathlib import Path

import pyvale.render as render
from _blender_example_tools import build_scene


def main() -> None:


    output_dir = Path.cwd() / "pyvale-output" / "render-blender-calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    _, camera, _ = build_scene()
    stereo = render.CameraTools.faceon_stereo_cameras(camera, 15.0)
    stereo.save_calibration(output_dir)
    print(render.calibration_image_count(render.BlenderCalibrationData()))


if not render.blender_available():
    print("Blender is unavailable: skipping this example.")
    print("Blender requires Python 3.13 and the optional "
          "'pyvale[blender]' dependency.")
else:
    main()
