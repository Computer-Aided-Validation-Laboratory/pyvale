# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Contract tests for the unified render public API."""

import numpy as np
import pytest
import riley
from scipy.spatial.transform import Rotation

from pyvale import render


def _convention(elem_type: riley.EElemType) -> dict[str, riley.ConnectConvention]:
    return {"connect1": riley.ConnectConvention(
        elem_type, riley.EConnectAxis.ROW, 0, riley.ENodeOrder.RILEY
    )}


def make_camera() -> render.Camera:
    """Create a small valid perspective camera."""
    return render.Camera(
        pixels_num=np.array((16, 16)),
        pixels_size=np.array((0.1, 0.1)),
        pos_world=np.array((0.0, 0.0, 2.0)),
        rot_world=Rotation.identity(),
        roi_cent_world=np.zeros(3),
        focal_length=1.0,
    )


def make_mesh(shader: object) -> render.Mesh3D:
    """Create a valid front-facing triangle mesh."""
    return render.Mesh3D(
        render.EElemType.TRI3,
        np.array(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        np.array((0, 1, 2)),
        shader,
    )


def test_render_utilities_are_available_at_the_render_package_level() -> None:
    """Public helpers require no utility class or nested module lookup."""
    functions = (
        render.average_subpixel_image,
        render.blender_camera_from_resolution,
        render.calculate_edge_function,
        render.cam_calc_leng_per_px,
        render.cam_calc_px_per_leng,
        render.cam_coverage_to_fov_scale,
        render.cam_fov_scale_to_coverage,
        render.cam_pos_frame_points,
        render.uv_calc_texture_px_per_leng,
        render.uv_map_planar_scaled,
        render.stereo_build_faceon,
        render.image_save,
    )

    assert all(callable(function) for function in functions)


def test_scene3d_keeps_mutable_scene_lists() -> None:
    """Scene collections remain the caller's ordinary mutable lists."""
    meshes: list[object] = []
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
        render.ELightType.POINT,
        np.zeros(3),
        np.array((0.0, 0.0, -1.0)),
        1.0,
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
        None,
        riley.FunctionShader(riley.FuncShaderBuiltin.constant),
    )

    with pytest.raises(render.RenderInputError, match="CONVENTION"):
        render.Riley(riley.RasterConfig()).verify_input(
            render.Scene3D([mesh], [make_camera()]),
        )


def test_meshes3d_from_simdata_normalises_displacement_layout() -> None:
    """SimData displacement fields become frame-major renderer displacements."""
    from pyvale.dataio import SimData

    sim_data = SimData(
        coords=np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        connect={"connect1": np.array(((0, 1, 2),))},
        node_vars={
            "x": np.array(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
            "y": np.zeros((3, 2)),
            "z": np.zeros((3, 2)),
        },
    )
    mesh = render.meshes3d_from_simdata(
        sim_data,
        _convention(riley.EElemType.TRI3),
        shaders={"connect1": object()},
        displacement_keys=("x", "y", "z"),
    )["connect1"]
    assert mesh.displacements is not None
    assert mesh.displacements.shape == (2, 3, 3)
    assert mesh.displacements[1, 0, 0] == 1.0


def test_meshes3d_from_simdata_extracts_a_volume_surface() -> None:
    """A render Mesh3D is always a surface, even from volume SimData."""
    from pyvale.dataio import SimData

    sim_data = SimData(
        coords=np.array(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        ),
        connect={"connect1": np.array(((0, 1, 2, 3),))},
    )

    mesh = render.meshes3d_from_simdata(
        sim_data,
        _convention(riley.EElemType.TET4),
        shaders={"connect1": object()},
    )["connect1"]

    assert mesh.element_type is render.EElemType.TRI3
    assert mesh.coords.shape == (4, 3)
    assert mesh.connectivity.shape == (4, 3)


def test_meshes3d_from_simdata_returns_one_mesh_per_block() -> None:
    """Every connectivity block is converted independently and keeps its key."""
    from pyvale.dataio import SimData

    sim_data = SimData(
        coords=np.array(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (1.0, 1.0, 0.0),
            )
        ),
        connect={
            "left": np.array(((0, 1, 2),)),
            "right": np.array(((1, 3, 2),)),
        },
    )
    convention = riley.ConnectConvention(
        riley.EElemType.TRI3,
        riley.EConnectAxis.ROW,
        0,
        riley.ENodeOrder.RILEY,
    )

    meshes = render.meshes3d_from_simdata(
        sim_data,
        {"left": convention, "right": convention},
    )

    assert tuple(meshes) == ("left", "right")
    np.testing.assert_array_equal(meshes["left"].connectivity, ((0, 1, 2),))
    np.testing.assert_array_equal(meshes["right"].connectivity, ((1, 3, 2),))


def test_meshes3d_from_simdata_requires_exact_block_configuration() -> None:
    """Explicit per-block configuration cannot silently omit a mesh block."""
    from pyvale.dataio import SimData

    sim_data = SimData(
        coords=np.zeros((3, 3)),
        connect={"surface": np.array(((0, 1, 2),))},
    )

    with pytest.raises(ValueError, match="Convention keys"):
        render.meshes3d_from_simdata(sim_data, {})
