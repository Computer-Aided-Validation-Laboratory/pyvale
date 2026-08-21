#!/usr/bin/env python3
"""Generate committed PixInt2D gold images.

Run ``python scripts/gengold_pxint2d.py --write`` to deliberately refresh the
small committed 32 by 32 reference images. The current committed golds were
cross-verified against RCC before this standalone generator was introduced.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import pyvale.data as dataset
import pyvale.render as render


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "tests/render/gold_pxint2d"


def load_quad9_affine() -> render.Mesh2D:
    """Load the copied RCC 32-pixel Quad9 affine fixture."""
    directory = dataset.pxint2d_single_element_path(
        "plate42_cam32_quad9_affine",
    )
    coords = np.loadtxt(directory / "coords.csv", delimiter=",")[:, :2]
    connect = np.loadtxt(
        directory / "connectivity.csv", delimiter=",", dtype=np.intp,
    )
    displacement_x = np.loadtxt(directory / "field_disp_x.csv", delimiter=",")
    displacement_y = np.loadtxt(directory / "field_disp_y.csv", delimiter=",")
    values = np.stack((displacement_x, displacement_y), axis=2).transpose(1, 0, 2)
    return render.Mesh2D(
        render.EElementType.QUAD9,
        coords,
        connect[None, :],
        values,
    )


def render_images() -> dict[str, np.ndarray]:
    """Render deterministic 32-pixel Grid2D and Speck2D references."""
    mesh = load_quad9_affine()
    camera = render.Camera2D(
        pixels_count=np.array((32, 32)), leng_per_px=1.0,
        roi_cent_world=np.zeros(3), subsample=1,
    )
    images: dict[str, np.ndarray] = {}
    for samples in (1, 2, 4):
        grid = render.PixIntGrid2D(
            options=render.PxInt2DOpts(
                mapping=render.EPxIntMapping.AFFINE,
                integration=render.RectRule(samples),
            ),
        )
        grid_result = grid.render(mesh, camera)
        images[f"affine_grid_rect{samples}.npy"] = grid_result.images[3, 0, :, :, 0]

        for kind, jitter_pdf, jitter, edge_fraction in (
            ("disk", "uniform", 0.25, 0.1),
            ("gaussian", "gaussian", 0.12, 0.4),
        ):
            pattern = render.AdditiveSpeckles.jittered_lattice(
                kind=kind, speckle_diameter=5.0, black_area_fraction=0.6,
                jitter_pdf=jitter_pdf, jitter=jitter, seed=3,
                bounds=(-20.0, 20.0, -20.0, 20.0),
                gaussian_edge_fraction=edge_fraction, tail_sigmas=8.0,
            )
            speck = render.PixIntSpeck2D(
                pattern,
                options=render.PxInt2DOpts(
                    mapping=render.EPxIntMapping.AFFINE,
                    integration=render.RectRule(samples),
                ),
            )
            speck_result = speck.render(mesh, camera)
            images[f"affine_speck_{kind}_rect{samples}.npy"] = (
                speck_result.images[3, 0, :, :, 0]
            )
    return images


def main() -> None:
    """Generate and verify deterministic PixInt2D gold images."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="write committed gold images",
    )
    arguments = parser.parse_args()
    images = render_images()
    if arguments.write:
        GOLD.mkdir(parents=True, exist_ok=True)
        for name, image in images.items():
            np.save(GOLD / name, image)
    else:
        parser.error("choose --write to refresh committed gold images")


if __name__ == "__main__":
    main()
