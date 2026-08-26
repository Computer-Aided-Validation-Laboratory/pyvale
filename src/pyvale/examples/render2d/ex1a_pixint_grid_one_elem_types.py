"""PixIntGrid2D one-element mappings
====================================

Render the same analytic eggbox through the Riley-ordered Tri3, Tri6, Quad4,
Quad8, and Quad9 element definitions using ``NEWTON_ONE_ELEM``.
"""


from pathlib import Path

import numpy as np

import pyvale.render as render


def make_mesh(element_type: render.EElementType) -> render.Mesh2D:
    """Create one Riley-ordered element enclosing the small camera view."""
    nodes = {
        render.EElementType.TRI3: ((-20, -20), (20, -20), (-20, 20)),
        render.EElementType.TRI6: ((-20, -20), (20, -20), (-20, 20),
                                   (0, -20), (0, 0), (-20, 0)),
        render.EElementType.QUAD4: ((-20, -20), (20, -20), (20, 20),
                                    (-20, 20)),
        render.EElementType.QUAD8: ((-20, -20), (20, -20), (20, 20),
                                    (-20, 20), (0, -20), (20, 0), (0, 20),
                                    (-20, 0)),
        render.EElementType.QUAD9: ((-20, -20), (20, -20), (20, 20),
                                    (-20, 20), (0, -20), (20, 0), (0, 20),
                                    (-20, 0), (0, 0)),
    }[element_type]
    coords = np.asarray(nodes, dtype=np.float64)
    return render.Mesh2D(
        element_type,
        coords,
        np.arange(len(coords))[None, :],
    )


camera = render.Camera2D(
    pixels_num=np.array((32, 32)), pixels_size=0.5,
    roi_cent_world=np.array((-8.0, -8.0, 0.0)),
)
renderer = render.PixIntGrid2D(
    options=render.PxInt2DOpts(
        mapping=render.EPxIntMapping.NEWTON_ONE_ELEM,
        integration=render.RectRule(2),
    ),
)
output_dir = Path.cwd() / "pyvale-output" / "render-pixint-grid-one-element"
output_dir.mkdir(parents=True, exist_ok=True)
for element_type in render.EElementType:
    mesh = make_mesh(element_type)
    scene = render.Scene2D(mesh=mesh, camera=camera)
    image = renderer.render(scene).images[0, 0, :, :, 0]
    np.save(output_dir / f"{element_type.value}.npy", image)
    print(element_type.value, image.shape, image.min(), image.max())
