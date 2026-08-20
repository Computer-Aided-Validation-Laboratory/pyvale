"""Generate the committed Blender triangle image regression array."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.render as render


def render_triangle(output_dir: Path) -> np.ndarray:
    """Render the deterministic common-API Blender triangle scene."""
    mesh = render.Mesh(
        render.EElementType.TRI3,
        np.array(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0),
                  (0.0, 1.0, 0.0))),
        np.array(((0, 1, 2),)), object(),
    )
    camera = render.Camera(
        np.array((32, 32)), np.array((0.02, 0.02)), np.array((0.0, 0.0, 2.0)),
        Rotation.identity(), np.zeros(3), 1.0,
    )
    result = render.Blender(render.BlenderConfig(output_dir, samples=1)).render(
        render.RenderScene((mesh,), (camera,)),
    )
    assert result.images is not None
    return result.images


def main() -> None:
    """Write the trusted Blender regression image."""
    path = Path(__file__).parent / "gold_blender/triangle.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, render_triangle(path.parent / "output"))


if __name__ == "__main__":
    main()
