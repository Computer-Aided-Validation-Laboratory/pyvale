# %%
"""Blender stereo deformation
==========================

Render every displacement frame for a convergent Blender stereo pair.
"""

from pathlib import Path

import pyvale.render as render
from _blender_example_tools import build_scene


mesh, camera, lights = build_scene()
stereo = render.CameraTools.faceon_stereo_cameras(camera, 15.0)
renderer = render.Blender(render.BlenderConfig(
    Path("pyvale-output/blender-stereo-deformation"), render_deformed=True,
))
result = renderer.render([mesh], stereo, lights)
assert result.images is not None
print(result.images.shape)
