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

import pyvale.verif.renderverif as renderverif


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "tests/render/gold_pxint2d"


def render_images() -> dict[str, np.ndarray]:
    """Render deterministic 32-pixel Grid2D and Speck2D references."""
    images: dict[str, np.ndarray] = {}
    for samples in (1, 2, 4):
        images[f"affine_grid_rect{samples}.npy"] = (
            renderverif.pxint2d_affine_reference(samples)
        )
        for kind in ("disk", "gaussian"):
            images[f"affine_speck_{kind}_rect{samples}.npy"] = (
                renderverif.pxint2d_affine_reference(samples, kind)
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
