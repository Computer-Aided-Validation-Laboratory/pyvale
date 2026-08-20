# %%
"""Blender stereo scene
====================

Render a convergent stereo pair with Blender through ``pyvale.render``.
"""

from pathlib import Path

import pyvale.render as render
from _blender_example_tools import build_scene


mesh, camera, lights = build_scene()
stereo = render.CameraTools.symmetric_stereo_cameras(camera, 15.0)
renderer = render.Blender(render.BlenderConfig(Path("pyvale-output/blender-stereo")))
result = renderer.render([mesh], stereo, lights)
assert result.images is not None
print(result.images.shape)
