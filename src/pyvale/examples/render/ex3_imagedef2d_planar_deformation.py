# %%
"""Planar ImageDef2D warp
=========================

Use the separate planar image-warp interface for a simple orthographic mesh.
"""

from pathlib import Path

import numpy as np

import pyvale.render as render


camera = render.Camera2D(
    pixels_count=np.array((32, 32)),
    leng_per_px=1.0,
    roi_cent_world=np.array((0.0, 0.0, 0.0)),
    subsample=1,
)
image = np.arange(32 * 32, dtype=np.float64).reshape(32, 32)
coords = np.array(((-16.0, -16.0), (16.0, -16.0),
                   (16.0, 16.0), (-16.0, 16.0)))
connectivity = np.array(((0, 1, 2, 3),))
displacements = np.zeros((1, 4, 2))
result = render.ImageDef2D().render(
    image, camera, coords, connectivity, displacements,
)
output_dir = Path.cwd() / "pyvale-output" / "render-imagedef2d"
output_dir.mkdir(parents=True, exist_ok=True)
np.save(output_dir / "warped_images.npy", result.images)
print(result.images.shape)
