# %%
"""PixIntGrid2D unstructured mesh mapping
=========================================

``NEWTON_MESH_UNSTRUCT`` accepts multiple Riley-ordered finite elements. This
small two-Quad4 example applies an affine displacement to both elements.
"""

import numpy as np

import pyvale.render as render


coords = np.array((
    (-16.0, -16.0), (0.0, -16.0), (16.0, -16.0),
    (-16.0, 16.0), (0.0, 16.0), (16.0, 16.0),
))
connectivity = np.array(((0, 1, 4, 3), (1, 2, 5, 4)))
mesh = render.Mesh2D(render.EElementType.QUAD4, coords, connectivity)
displacements = np.stack((
    np.zeros((len(coords), 2)),
    np.column_stack((0.02 * coords[:, 0], -0.01 * coords[:, 1])),
))
camera = render.Camera2D(
    pixels_count=np.array((32, 32)), leng_per_px=1.0,
    roi_cent_world=np.zeros(3),
)
renderer = render.PixIntGrid2D(
    options=render.PxInt2DOpts(
        mapping=render.EPxIntMapping.NEWTON_MESH_UNSTRUCT,
        integration=render.GaussRule(2),
    ),
)
result = renderer.render(mesh, camera, render.DisplacementSeries2D(displacements))
print(result.images.shape, result.masks.all())
