# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Smoke test for the Blender unified-renderer adapter."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.render as render


def test_blender_adapter_renders_common_mesh_and_camera(tmp_path: Path) -> None:
    """The Blender adapter accepts render.Mesh and normalises its image result."""
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
    result = render.Blender(render.BlenderConfig(tmp_path, samples=1)).render(
        [mesh], [camera],
    )
    assert result.images is not None
    assert result.images.shape == (1, 1, 32, 32, 1)
