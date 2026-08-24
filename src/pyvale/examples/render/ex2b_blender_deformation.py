"""Blender deformation
===================

Render all displacement frames of a finite-element surface in Blender.
"""


from pathlib import Path

import pyvale.render as render
from _blender_example_tools import build_scene


def main() -> None:


    renderer = render.Blender(render.BlenderConfig(
        Path.cwd() / "pyvale-output" / "render-blender-deformation",
        render_deformed=True,
    ))
    mesh, camera, lights = build_scene()
    result = renderer.render(render.RenderScene((mesh,), (camera,), tuple(lights)))
    assert result.images is not None
    print(f"Rendered {result.images.shape[0]} deformation frames.")


if not render.blender_available():
    print("Blender is unavailable: skipping this example.")
    print("Blender requires Python 3.13 and the optional "
          "'pyvale[blender]' dependency.")
else:
    main()
