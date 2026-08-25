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


def make_mesh(shader: object) -> render.Mesh3D:
    """Create a valid front-facing triangle mesh."""
    return render.Mesh3D(
        render.EElementType.TRI3,
        np.array(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        np.array(((0, 1, 2))), shader,
    )


def test_camera2d_defaults_to_one_sample_per_pixel() -> None:
    """The planar and perspective camera defaults use one sample per pixel."""
    assert render.Camera2D().subsample == 1


def test_scene3d_keeps_mutable_scene_lists() -> None:
    """Scene collections remain the caller's ordinary mutable lists."""
    meshes: list[render.RenderMesh] = []
    cameras = [make_camera()]

    scene = render.Scene3D(meshes, cameras)

    assert scene.meshes is meshes
    assert scene.cameras is cameras


class _FakeRenderer(render.IRenderer3D):
    """Renderer spy used to enforce the template-method lifecycle."""

    def __init__(self) -> None:
        self.rendered = False

    def verify_input(self, scene: render.Scene3D) -> None:
        if not scene.meshes:
            raise render.RenderInputError(())

    def _render(self, scene: render.Scene3D):
        self.rendered = True
        return render.RenderResult(np.zeros((1, 1, 1, 1, 1)))


def test_renderer_lifecycle_blocks_backend_after_validation_failure() -> None:
    """The ABC must never invoke backend work when verification fails."""
    renderer = _FakeRenderer()
    with pytest.raises(render.RenderInputError):
        renderer.render(render.Scene3D([], [make_camera()]))
    assert not renderer.rendered


def test_riley_rejects_lights_before_backend_call(monkeypatch) -> None:
    """Unsupported lights fail before Riley mesh/camera conversion or rasterising."""
    import pyvale.render.riley as riley_adapter

    renderer = render.Riley(riley_adapter.riley.RasterConfig())
    light = render.Light(
        render.ELightType.POINT, np.zeros(3), np.array((0.0, 0.0, -1.0)), 1.0,
    )
    with pytest.raises(render.RenderInputError, match="UNSUPPORTED"):
        renderer.render(render.Scene3D([], [make_camera()], [light]))


def test_riley_rejects_meshes_outside_the_shared_convention() -> None:
    """Native Riley input must obey the common winding convention."""
    import riley

    mesh = riley.Mesh(
        riley.MeshType.tri3,
        np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        np.array(((0, 2, 1),)),
    )

    with pytest.raises(render.RenderInputError, match="CONVENTION"):
        render.Riley(riley.RasterConfig()).verify_input(
            render.Scene3D([mesh], [make_camera()]),
        )


def test_mesh3d_from_simdata_normalises_displacement_layout() -> None:
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
    mesh = render.mesh3d_from_simdata(
        sim_data, object(), displacement_keys=("x", "y", "z"),
    )
    assert mesh.displacements is not None
    assert mesh.displacements.shape == (2, 3, 3)
    assert mesh.displacements[1, 0, 0] == 1.0


def test_mesh3d_from_simdata_extracts_a_volume_surface() -> None:
    """A render Mesh3D is always a surface, even from volume SimData."""
    from pyvale.dataio import SimData

    sim_data = SimData(
        coords=np.array((
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )),
        connect={"connect1": np.array(((0, 1, 2, 3),))},
    )

    mesh = render.mesh3d_from_simdata(sim_data, object())

    assert mesh.element_type is render.EElementType.TRI3
    assert mesh.coords.shape == (4, 3)
    assert mesh.connectivity.shape == (4, 3)


def test_mesh2d_from_simdata_uses_xy_displacements() -> None:
    """The 2D converter makes an XY mesh with frame-major displacement."""
    from pyvale.dataio import SimData

    sim_data = SimData(
        coords=np.array((
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )),
        connect={"connect1": np.array(((0, 1, 2),))},
        node_vars={
            "x": np.array(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
            "y": np.zeros((3, 2)),
        },
    )

    mesh = render.mesh2d_from_simdata(sim_data, ("x", "y"))

    assert mesh.coords.shape == (3, 2)
    assert mesh.displacement.shape == (2, 3, 2)
    assert mesh.displacement[1, 0, 0] == 1.0


def test_mesh2d_from_simdata_rejects_non_xy_meshes() -> None:
    """The first Mesh2D conversion API intentionally supports XY only."""
    from pyvale.dataio import SimData

    sim_data = SimData(
        coords=np.array((
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        )),
        connect={"connect1": np.array(((0, 1, 2),))},
    )

    with pytest.raises(ValueError, match="XY plane"):
        render.mesh2d_from_simdata(sim_data)
