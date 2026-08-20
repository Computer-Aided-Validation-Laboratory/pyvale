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
output_dir = Path.cwd() / "pyvale-output" / "render-blender-stereo"
renderer = render.Blender(render.BlenderConfig(output_dir))
result = renderer.render(render.RenderScene(
    (mesh,), (stereo.cam_data_0, stereo.cam_data_1), tuple(lights),
))
assert result.images is not None
print(result.images.shape)
