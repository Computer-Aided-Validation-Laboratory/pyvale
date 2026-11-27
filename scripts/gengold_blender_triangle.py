"""Generate the committed Blender triangle image regression array."""

from pathlib import Path

import numpy as np

import pyvale.verif.renderverif as renderverif


def main() -> None:
    """Write the trusted Blender regression image."""
    repository_root = Path(__file__).resolve().parents[1]
    path = repository_root / "tests" / "render" / "gold_blender" / "triangle.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, renderverif.render_triangle(path.parent / "output"))


if __name__ == "__main__":
    main()
