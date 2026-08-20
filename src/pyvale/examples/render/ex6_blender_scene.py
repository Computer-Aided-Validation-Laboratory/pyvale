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
renderer = render.Blender(render.BlenderConfig(Path("pyvale-output/blender")))
result = renderer.render([mesh], [camera], lights)
assert result.images is not None
print(result.images.shape)
