# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Contract tests for the unified render public API."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import pyvale.render as render


def make_camera() -> render.Camera:
    """Create a small valid perspective camera."""
    return render.Camera(
        np.array((16, 16)), np.array((0.1, 0.1)), np.array((0.0, 0.0, 2.0)),
        Rotation.identity(), np.zeros(3), 1.0,
    )


def make_mesh(shader: object) -> render.Mesh:
    """Create a valid front-facing triangle mesh."""
    return render.Mesh(
        render.EElementType.TRI3,
        np.array(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        np.array(((0, 1, 2))), shader,
    )


class _FakeRenderer(render.IRenderer3D):
    """Renderer spy used to enforce the template-method lifecycle."""

    def __init__(self) -> None:
        self.rendered = False

    def verify_input(self, meshes, cameras, lights=None):
        if not meshes:
            raise render.RenderInputError(())
        return None

    def _render(self, render_plan):
        self.rendered = True
        return render.RenderResult(np.zeros((1, 1, 1, 1, 1)))


def test_renderer_lifecycle_blocks_backend_after_validation_failure() -> None:
    """The ABC must never invoke backend work when verification fails."""
    renderer = _FakeRenderer()
    with pytest.raises(render.RenderInputError):
        renderer.render([], [make_camera()])
    assert not renderer.rendered


def test_riley_rejects_lights_before_backend_call(monkeypatch) -> None:
    """Unsupported lights fail before Riley mesh/camera conversion or rasterising."""
    import pyvale.render.riley as riley_adapter

    class RasterConfig:
        pass

    class FakeRiley:
        pass

    FakeRiley.RasterConfig = RasterConfig

    monkeypatch.setattr(riley_adapter, "_riley", FakeRiley)
    renderer = render.Riley(RasterConfig())
    light = render.Light(
        render.ELightType.POINT, np.zeros(3), np.array((0.0, 0.0, -1.0)), 1.0,
    )
    with pytest.raises(render.RenderInputError, match="UNSUPPORTED"):
        renderer.render([make_mesh(object())], [make_camera()], [light])


def test_mesh_from_simdata_normalises_displacement_layout() -> None:
    """SimData displacement fields become frame-major renderer displacements."""
    from pyvale.dataio import SimData

    sim_data = SimData(
        coords=np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                         (0.0, 1.0, 0.0))),
        connect={"connect1": np.array(((0, 1, 2),))},
        node_vars={
            "x": np.array(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
            "y": np.zeros((3, 2)),
            "z": np.zeros((3, 2)),
        },
    )
    mesh = render.mesh_from_simdata(
        sim_data, object(), displacement_keys=("x", "y", "z"),
    )
    assert mesh.displacements is not None
    assert mesh.displacements.shape == (2, 3, 3)
    assert mesh.displacements[1, 0, 0] == 1.0
