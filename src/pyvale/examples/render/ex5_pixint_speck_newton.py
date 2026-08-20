# %%
"""PixIntSpeck2D with a Newton mesh map
=======================================

Create a deterministic analytic disk-speckle image over a Riley-ordered Quad9
element. Rectangular quadrature controls the sub-pixel sampling density.
"""

import numpy as np
from pathlib import Path

import pyvale.render as render


coords = np.array((
    (-16.0, -16.0), (16.0, -16.0), (16.0, 16.0), (-16.0, 16.0),
    (0.0, -16.0), (16.0, 0.0), (0.0, 16.0), (-16.0, 0.0), (0.0, 0.0),
))
mesh = render.Mesh2D(
    render.EElementType.QUAD9, coords, np.arange(9)[None, :],
)
pattern = render.AdditiveSpeckles.jittered_lattice(
    kind="disk", speckle_diameter=2.0, black_area_fraction=0.5,
    jitter_pdf="uniform", jitter=0.1, seed=3,
    bounds=(-20.0, 20.0, -20.0, 20.0),
)
camera = render.Camera2D(
    pixels_count=np.array((32, 32)), leng_per_px=1.0,
    roi_cent_world=np.zeros(3),
)
result = render.PixIntSpeck2D(
    pattern,
    options=render.PxInt2DOpts(
        mapping=render.EPxIntMapping.NEWTON_MESH_UNSTRUCT,
        integration=render.RectRule(4),
    ),
).render(mesh, camera, render.DisplacementSeries2D(np.zeros((1, 9, 2))))
output_dir = Path.cwd() / "pyvale-output" / "render-pixint-speck"
output_dir.mkdir(parents=True, exist_ok=True)
np.save(output_dir / "warped_images.npy", result.images)
print(result.images.shape, result.images.min(), result.images.max())
