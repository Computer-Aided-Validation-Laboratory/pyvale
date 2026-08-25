"""Blender stereo deformation
==========================

Render every displacement frame for a convergent Blender stereo pair.
"""


from pathlib import Path

import pyvale.render as render
from _blender_example_tools import build_scene


def main() -> None:


    mesh, camera, lights = build_scene()
    stereo = render.CameraTools.faceon_stereo_cameras(camera, 15.0)
    renderer = render.Blender(render.BlenderConfig(
        Path.cwd() / "pyvale-output" / "render-blender-stereo-deformation",
        render_deformed=True,
    ))
    result = renderer.render(render.Scene3D(
        [mesh], [stereo.cam_data_0, stereo.cam_data_1], lights,
    ))
    assert result.images is not None
    print(result.images.shape)


if not render.blender_available():
    print("Blender is unavailable: skipping this example.")
    print("Blender requires Python 3.13 and the optional "
          "'pyvale[blender]' dependency.")
else:
    main()
