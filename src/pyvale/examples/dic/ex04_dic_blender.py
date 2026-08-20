# %%
"""Blender-rendered DIC
====================

Render a deforming finite-element surface with ``render.Blender`` and correlate
the reference and first deformed images without an interactive ROI window.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.dataset as dataset
import pyvale.dic as dic
import pyvale.mooseherder as mooseherder
import pyvale.render as render
import pyvale.sensorsim as sensorsim


output_dir = Path("pyvale-output/dic-blender")
sim_data = mooseherder.ExodusLoader(dataset.mechanical_2d_path()).load_all_sim_data()
sensorsim.scale_length_units(1000.0, sim_data, ("disp_x", "disp_y"))
camera = render.Camera(
    np.array((64, 64)), np.array((0.00345, 0.00345)),
    np.array((0.0, 0.0, 500.0)), Rotation.identity(), np.zeros(3), 15.0,
)
texture_resolution = camera.pixels_size[0] * 500.0 / camera.focal_length
mesh = render.blender.mesh_from_simdata(
    sim_data,
    render.BlenderTextureShader(dataset.dic_pattern_5mpx_path(), texture_resolution),
    ("disp_x", "disp_y"),
)
images = render.Blender(render.BlenderConfig(
    output_dir / "render", threads=1, render_deformed=True,
)).render(
    render.RenderScene(
        (mesh,),
        (camera,),
        (render.Light(
            render.ELightType.POINT, np.array((0.0, 0.0, 400.0)),
            np.zeros(3), 1.0,
        ),),
    ),
).images
assert images is not None

reference = images[0, 0, :, :, 0]
deformed = images[1, 0, :, :, 0]
roi_mask = np.ones(reference.shape, dtype=bool)
dic.calculate_2d(
    reference=reference,
    deformed=deformed,
    roi_mask=roi_mask,
    seed=[reference.shape[1] // 2, reference.shape[0] // 2],
    subset_size=21,
    subset_step=16,
    max_displacement=8,
    output_basepath=output_dir,
    output_prefix="blender_dic_",
)
print(f"DIC results written to {output_dir}.")
