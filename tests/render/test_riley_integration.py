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
import riley

from scripts.gengold_riley_rabbits import build_rabbit_meshes
from render_checks import assert_render_allclose


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
        np.array((32, 32)), np.array((0.02, 0.02)), np.array((0.0, 0.0, 2.0)),
        Rotation.identity(), np.zeros(3), 1.0,
    )
    config = riley.create_raster_config(1, save_strategy=riley.SaveStrategy.memory)
    result = render.Riley(config).render(render.RenderScene((mesh,), (camera,)))
    assert result.images is not None
    assert result.images.shape == (1, 1, 32, 32, 1)


def test_riley_rabbit_multimesh_golden_regression() -> None:
    """The Riley rabbit multi-mesh scene remains deterministic end-to-end."""
    meshes = build_rabbit_meshes()
    coords = np.concatenate([mesh.coords for mesh in meshes])
    pixels_num = np.array((320, 160))
    pixels_size = np.array((5.3e-6, 5.3e-6))
    focal_length = 50.0e-3
    rotation = Rotation.identity()
    position = riley.pos_fill_frame_from_rot(
        coords, tuple(pixels_num), tuple(pixels_size), focal_length,
        tuple(rotation.as_euler("xyz")), 1.1,
    )
    camera = render.Camera(
        pixels_num, pixels_size, np.asarray(position), rotation,
        np.mean(coords, axis=0), focal_length,
    )
    config = riley.create_raster_config(1, save_strategy=riley.SaveStrategy.memory)
    result = render.Riley(config).render(render.RenderScene(tuple(meshes), (camera,)))
    assert result.images is not None
    golden_path = Path(__file__).parent / "gold_riley" / "rabbits.npy"
    assert_render_allclose(
        result.images, np.load(golden_path), "riley_rabbit_multimesh",
        rtol=0.0, atol=0.0,
    )
