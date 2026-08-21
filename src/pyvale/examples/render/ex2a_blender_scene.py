# %%
"""Blender scene
===============

Render a textured finite-element surface with Blender through the unified
``pyvale.render`` API.
"""

from pathlib import Path

import pyvale.render as render
from _blender_example_tools import build_scene


mesh, camera, lights = build_scene()
output_dir = Path.cwd() / "pyvale-output" / "render-blender-scene"
renderer = render.Blender(render.BlenderConfig(output_dir))
result = renderer.render(render.RenderScene((mesh,), (camera,), tuple(lights)))
assert result.images is not None
print(result.images.shape)
