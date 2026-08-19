# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Tests for Blender's optional-backend boundary."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import pyvale.render as render
import pyvale.render.blender as blender_adapter


def make_camera() -> render.Camera:
    """Create a small valid perspective camera."""
    return render.Camera(
        np.array((16, 16)),
        np.array((0.1, 0.1)),
        np.array((0.0, 0.0, 2.0)),
        Rotation.identity(),
        np.zeros(3),
        1.0,
    )


def make_mesh() -> render.Mesh:
    """Create a small valid triangle mesh."""
    return render.Mesh(
        render.EElementType.TRI3,
        np.array(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        np.array(((0, 1, 2),)),
        object(),
    )


def test_blender_reports_unavailable_before_scene_construction(
    monkeypatch,
    tmp_path,
) -> None:
    """An unavailable optional backend is a render validation error."""
    reason = "Blender requires Python 3.13 and the blender extra."
    monkeypatch.setattr(
        blender_adapter,
        "_blender_unavailable_reason",
        lambda: reason,
    )
    renderer = render.Blender(render.BlenderConfig(tmp_path))

    with pytest.raises(render.RenderInputError, match="UNAVAILABLE"):
        renderer.render((make_mesh(),), (make_camera(),))


def test_blender_available_reflects_backend_probe(monkeypatch) -> None:
    """The public availability helper uses the same preflight probe."""
    monkeypatch.setattr(
        blender_adapter,
        "_blender_unavailable_reason",
        lambda: None,
    )

    assert blender_adapter.blender_available()
