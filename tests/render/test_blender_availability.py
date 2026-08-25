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
import pyvale.render.blender.adapter as blender_adapter


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


def make_mesh() -> render.Mesh3D:
    """Create a small valid triangle mesh."""
    return render.Mesh3D(
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
        renderer.render(render.Scene3D([make_mesh()], [make_camera()]))


def test_blender_available_reflects_backend_probe(monkeypatch) -> None:
    """The public availability helper uses the same preflight probe."""
    monkeypatch.setattr(
        blender_adapter,
        "_blender_unavailable_reason",
        lambda: None,
    )

    assert blender_adapter.blender_available()


def test_blender_gpu_probe_is_safe_when_backend_is_unavailable(monkeypatch) -> None:
    """The legacy GPU capability query returns false without Blender."""
    monkeypatch.setattr(
        blender_adapter,
        "_blender_unavailable_reason",
        lambda: "Blender is unavailable.",
    )

    assert not blender_adapter.blender_gpu_available()


def test_blender_warns_for_non_tri3_meshes(monkeypatch, tmp_path) -> None:
    """Blender retains legacy meshes but signals its Tri3-only guarantee."""
    monkeypatch.setattr(
        blender_adapter,
        "_blender_unavailable_reason",
        lambda: None,
    )
    mesh = render.Mesh3D(
        render.EElementType.QUAD4,
        np.array((
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
        )),
        np.array(((0, 1, 2, 3),)),
        object(),
    )

    with pytest.warns(RuntimeWarning, match="Tri3"):
        render.Blender(render.BlenderConfig(tmp_path)).verify_input(
            render.Scene3D([mesh], [make_camera()]),
        )
