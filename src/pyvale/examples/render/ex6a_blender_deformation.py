# %%
"""Blender deformation
===================

Render all displacement frames of a finite-element surface in Blender.
"""

from pathlib import Path

import pyvale.render as render
from _blender_example_tools import build_scene


renderer = render.Blender(render.BlenderConfig(
    Path.cwd() / "pyvale-output" / "render-blender-deformation",
    render_deformed=True,
))
mesh, camera, lights = build_scene()
result = renderer.render(render.RenderScene((mesh,), (camera,), tuple(lights)))
assert result.images is not None
print(f"Rendered {result.images.shape[0]} deformation frames.")
