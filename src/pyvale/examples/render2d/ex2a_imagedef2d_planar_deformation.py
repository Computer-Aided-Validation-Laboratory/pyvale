"""Planar ImageDef2D warp
=========================

Use the separate planar image-warp interface for a simple orthographic mesh.
"""

# NOTE: This example uses ImageDef2D, which is provided for legacy
# compatibility only. Prefer the Riley renderer even for 2D rendering: it is
# faster and more deeply verified for correctness.

from pathlib import Path

import numpy as np

import pyvale.render as render


camera = render.Camera2D(
    pixels_count=np.array((32, 32)),
    pixel_size=1.0,
    roi_cent_world=np.array((0.0, 0.0, 0.0)),
    subsample=1,
)
image = np.arange(32 * 32, dtype=np.float64).reshape(32, 32)
coords = np.array(((-16.0, -16.0), (16.0, -16.0),
                   (16.0, 16.0), (-16.0, 16.0)))
connectivity = np.array(((0, 1, 2, 3),))
displacements = np.zeros((1, 4, 2))

mesh = render.Mesh2D(
    element_type=render.EElementType.QUAD4,
    coords=coords,
    connectivity=connectivity,
    displacement=displacements,
)

scene = render.Scene2D(
    mesh=mesh,
    camera=camera,
    source_image=image,
)

result = render.ImageDef2D().render(scene)
output_dir = Path.cwd() / "pyvale-output" / "render-imagedef2d"
output_dir.mkdir(parents=True, exist_ok=True)
np.save(output_dir / "warped_images.npy", result.images)
print(result.images.shape)
