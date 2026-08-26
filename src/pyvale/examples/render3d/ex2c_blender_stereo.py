"""Blender stereo scene
====================

Render a convergent stereo pair with Blender through ``pyvale.render``.
"""


from pathlib import Path

import pyvale.render as render
from _blender_example_tools import build_scene


def main() -> None:


    mesh, camera, lights = build_scene()
    stereo = render.symmetric_stereo_cameras(camera, 15.0)
    output_dir = Path.cwd() / "pyvale-output" / "render-blender-stereo"
    renderer = render.Blender(render.BlenderConfig(output_dir))
    result = renderer.render(render.Scene3D(
        [mesh], [stereo.camera_0, stereo.camera_1], lights,
    ))
    assert result.images is not None
    print(result.images.shape)


if not render.blender_available():
    print("Blender is unavailable: skipping this example.")
    print("Blender requires Python 3.13 and the optional "
          "'pyvale[blender]' dependency.")
else:
    main()
