# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Real Riley integration and golden-regression tests."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.render as render
import pyvale.verif.renderverif as renderverif
import riley

from pyvale.verif.renderverif import assert_render_allclose


def test_common_camera_reorders_scipy_angles_for_riley() -> None:
    """Riley receives its native Z-Y-X angle storage order."""
    camera = render.Camera(
        pixels_num=np.array((32, 32)),
        pixels_size=np.array((0.02, 0.02)),
        pos_world=np.array((0.0, 0.0, 2.0)),
        rot_world=Rotation.from_euler(
            "xyz",
            (10.0, 20.0, 30.0),
            degrees=True,
        ),
        roi_cent_world=np.zeros(3),
        focal_length=1.0,
    )

    native = render.to_riley_camera(camera)

    np.testing.assert_allclose(
        native.rot_world,
        np.radians((30.0, 20.0, 10.0)),
    )


def test_riley_returns_canonical_image_layout() -> None:
    """Riley's field-major buffer is normalised by the pyvale adapter."""
    mesh = riley.Mesh(
        riley.MeshType.tri3,
        np.array(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        np.array(((0, 1, 2),)),
        shader_type=riley.ShaderType.func,
        func_shader_builtin=riley.FuncShaderBuiltin.constant,
        func_shader_coord_mode=riley.FuncCoordMode.world_reference,
    )
    camera = render.Camera(
        pixels_num=np.array((32, 32)),
        pixels_size=np.array((0.02, 0.02)),
        pos_world=np.array((0.0, 0.0, 2.0)),
        rot_world=Rotation.identity(),
        roi_cent_world=np.zeros(3),
        focal_length=1.0,
    )
    config = riley.create_raster_config(
        1, save_strategy=riley.SaveStrategy.memory
    )
    result = render.Riley(config).render(render.Scene3D([mesh], [camera]))
    assert result.images is not None
    assert result.images.shape == (1, 1, 32, 32, 1)


def test_common_mesh_matches_native_riley_mesh() -> None:
    """Mesh3D conversion preserves native Riley rendering exactly."""
    coords = np.array(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)))
    connectivity = np.array(((0, 1, 2),))
    native_mesh = riley.Mesh(
        riley.MeshType.tri3,
        coords,
        connectivity,
        shader_type=riley.ShaderType.func,
        func_shader_builtin=riley.FuncShaderBuiltin.checker,
        func_shader_coord_mode=riley.FuncCoordMode.world_reference,
    )
    common_mesh = render.Mesh3D(
        element_type=render.EElementType.TRI3,
        coords=coords,
        connectivity=connectivity,
        shader=render.RileyFunctionShader(
            builtin=riley.FuncShaderBuiltin.checker,
            coord_mode=riley.FuncCoordMode.world_reference,
        ),
    )
    camera = render.Camera(
        pixels_num=np.array((32, 32)),
        pixels_size=np.array((0.02, 0.02)),
        pos_world=np.array((0.0, 0.0, 2.0)),
        rot_world=Rotation.identity(),
        roi_cent_world=np.zeros(3),
        focal_length=1.0,
    )

    def render_mesh(mesh: object) -> np.ndarray:
        config = riley.create_raster_config(
            1,
            total_threads=1,
            save_strategy=riley.SaveStrategy.memory,
        )
        result = render.Riley(config).render(render.Scene3D([mesh], [camera]))
        assert result.images is not None
        return result.images

    np.testing.assert_array_equal(
        render_mesh(common_mesh),
        render_mesh(native_mesh),
    )


def test_riley_rabbit_multimesh_golden_regression() -> None:
    """The Riley rabbit multi-mesh scene remains deterministic end-to-end."""
    result = render.Riley(renderverif.riley_memory_config()).render(
        renderverif.riley_rabbit_scene(),
    )
    assert result.images is not None
    golden_path = Path(__file__).parent / "gold_riley" / "rabbits.npy"
    assert_render_allclose(
        result.images,
        np.load(golden_path),
        "riley_rabbit_multimesh",
        rtol=0.0,
        atol=0.0,
    )
