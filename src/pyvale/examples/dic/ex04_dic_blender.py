"""Blender-rendered DIC
====================

Render a deforming finite-element surface with ``render.Blender`` and correlate
the reference and first deformed images without an interactive ROI window.
"""


from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pyvale.dataio import SimData
import pyvale.data as dataset
import pyvale.dic as dic
import pyvale.render as render


def main() -> None:


    output_dir = Path("pyvale-output/dic-blender")
    # %% Intended render workflow: convert SimData to a surface Mesh3D.
    # The converter enforces the shared convention and skins volumes. Blender uses
    # Tri3 surface meshes, so this example supplies a Tri3 simulation mesh.
    sim_data = SimData(
        coords=np.array((
            (-100.0, -100.0, 0.0),
            (200.0, -100.0, 0.0),
            (-100.0, 200.0, 0.0),
        )),
        connect={"connect1": np.array(((0, 1, 2),))},
        node_vars={
            "disp_x": np.array(((0.0, 0.3), (0.0, 0.3), (0.0, 0.3))),
            "disp_y": np.zeros((3, 2)),
        },
    )
    camera = render.Camera(
        np.array((64, 64)), np.array((0.00345, 0.00345)),
        np.array((0.0, 0.0, 500.0)), Rotation.identity(), np.zeros(3), 15.0,
    )
    texture_resolution = camera.pixels_size[0] * 500.0 / camera.focal_length
    mesh = render.mesh3d_from_simdata(
        sim_data,
        render.BlenderTextureShader(
            dataset.dic_pattern_5mpx_path(),
            texture_resolution,
        ),
        ("disp_x", "disp_y"),
    )
    images = render.Blender(render.BlenderConfig(
        output_dir / "render", threads=1, render_deformed=True,
    )).render(
        render.Scene3D(
            [mesh],
            [camera],
            [render.Light(
                render.ELightType.POINT, np.array((0.0, 0.0, 400.0)),
                np.zeros(3), 1.0,
            )],
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


if not render.blender_available():
    print("Blender is unavailable: skipping this example.")
    print("Blender requires Python 3.13 and the optional "
          "'pyvale[blender]' dependency.")
else:
    main()
