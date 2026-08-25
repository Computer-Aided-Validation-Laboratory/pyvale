"""Blender scene
===============

Render a textured finite-element surface with Blender through the unified
``pyvale.render`` API.
"""


from pathlib import Path

import pyvale.render as render
from _blender_example_tools import build_scene


def main() -> None:


    mesh, camera, lights = build_scene()
    output_dir = Path.cwd() / "pyvale-output" / "render-blender-scene"
    renderer = render.Blender(render.BlenderConfig(output_dir))
    result = renderer.render(render.Scene3D([mesh], [camera], lights))
    assert result.images is not None
    print(result.images.shape)


if not render.blender_available():
    print("Blender is unavailable: skipping this example.")
    print("Blender requires Python 3.13 and the optional "
          "'pyvale[blender]' dependency.")
else:
    main()
