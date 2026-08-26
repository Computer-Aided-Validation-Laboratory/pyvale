"""PixIntGrid2D unstructured mesh mapping
=========================================

``NEWTON_MESH_UNSTRUCT`` accepts multiple Riley-ordered finite elements. This
small two-Quad4 example applies an affine displacement to both elements.
"""


from pathlib import Path

import numpy as np

import pyvale.render as render


coords = np.array((
    (-16.0, -16.0), (0.0, -16.0), (16.0, -16.0),
    (-16.0, 16.0), (0.0, 16.0), (16.0, 16.0),
))
connectivity = np.array(((0, 1, 4, 3), (1, 2, 5, 4)))
displacements = np.stack((
    np.zeros((len(coords), 2)),
    np.column_stack((0.02 * coords[:, 0], -0.01 * coords[:, 1])),
))
mesh = render.Mesh2D(
    render.EElementType.QUAD4,
    coords,
    connectivity,
    displacements,
)
camera = render.Camera2D(
    pixels_num=np.array((32, 32)), pixels_size=1.0,
    roi_cent_world=np.zeros(3),
)
renderer = render.PixIntGrid2D(
    options=render.PxInt2DOpts(
        mapping=render.EPxIntMapping.NEWTON_MESH_UNSTRUCT,
        integration=render.GaussRule(2),
    ),
)
scene = render.Scene2D(mesh=mesh, camera=camera)
result = renderer.render(scene)
output_dir = Path.cwd() / "pyvale-output" / "render-pixint-grid-mesh"
output_dir.mkdir(parents=True, exist_ok=True)
np.save(output_dir / "warped_images.npy", result.images)
print(result.images.shape, result.masks.all())
