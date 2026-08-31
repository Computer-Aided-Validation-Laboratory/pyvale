# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Smoke test for the Blender unified-renderer adapter."""

from pathlib import Path

import numpy as np
import pytest

import pyvale.render as render
import pyvale.verif.renderverif as renderverif

from pyvale.verif.renderverif import assert_render_allclose


pytestmark = pytest.mark.skipif(
    not render.blender_available(),
    reason="Blender requires Python 3.13 and the pyvale blender extra.",
)


def test_blender_adapter_renders_common_mesh_and_camera(tmp_path: Path) -> None:
    """The Blender adapter accepts render.Mesh3D and normalises its result."""
    actual = renderverif.render_triangle(tmp_path)
    reference = np.load(Path(__file__).parent / "gold_blender/triangle.npy")
    assert actual.shape == (1, 1, 32, 32, 1)
    assert_render_allclose(actual, reference, "blender_triangle", rtol=0.0, atol=0.0)
