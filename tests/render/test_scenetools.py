# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Analytic tests for multi-object scene placement and spatial layout."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import pyvale.render as render


def _make_unit_box(center: tuple[float, float, float]) -> render.Mesh3D:
    cx, cy, cz = center
    coords = np.array([
        [cx - 0.5, cy - 0.5, cz - 0.5],
        [cx + 0.5, cy - 0.5, cz - 0.5],
        [cx + 0.5, cy + 0.5, cz + 0.5],
    ])
    return render.Mesh3D(
        element_type=render.EElementType.TRI3,
        coords=coords,
        connectivity=np.array([[0, 1, 2]], dtype=np.uintp),
        shader=None,
    )


def test_scene_bounds_and_center() -> None:
    """Scene bounds compute componentwise min/max over all meshes."""
    m1 = _make_unit_box((-2.0, 0.0, 0.0))  # x in [-2.5, -1.5]
    m2 = _make_unit_box((3.0, 0.0, 0.0))   # x in [2.5, 3.5]

    lower, upper = render.scene_bounds([m1, m2])
    np.testing.assert_allclose(lower, np.array([-2.5, -0.5, -0.5]))
    np.testing.assert_allclose(upper, np.array([3.5, 0.5, 0.5]))

    sc = render.scene_center([m1, m2])
    np.testing.assert_allclose(sc, np.array([0.5, 0.0, 0.0]))


def test_scene_translate_and_rotate() -> None:
    """Scene transforms move all meshes while preserving relative distances."""
    m1 = _make_unit_box((-1.0, 0.0, 0.0))
    m2 = _make_unit_box((1.0, 0.0, 0.0))

    translated = render.scene_translate([m1, m2], [5.0, 10.0, 0.0])
    np.testing.assert_allclose(
        render.mesh_center(translated[0]), np.array([4.0, 10.0, 0.0])
    )
    np.testing.assert_allclose(
        render.mesh_center(translated[1]), np.array([6.0, 10.0, 0.0])
    )

    rot = Rotation.from_euler("z", 90.0, degrees=True)
    rotated = render.scene_rotate([m1, m2], rot, pivot=[0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        render.mesh_center(rotated[0]), np.array([0.0, -1.0, 0.0]), atol=1.0e-12
    )
    np.testing.assert_allclose(
        render.mesh_center(rotated[1]), np.array([0.0, 1.0, 0.0]), atol=1.0e-12
    )


def test_scene_arrange_points() -> None:
    """Placing meshes at explicit points aligns bounding-box centers."""
    meshes = [_make_unit_box((0, 0, 0)) for _ in range(3)]
    targets = [[10.0, 0.0, 0.0], [20.0, 5.0, 0.0], [30.0, -5.0, 0.0]]

    arranged = render.scene_arrange_points(meshes, targets)
    for m, target in zip(arranged, targets, strict=True):
        np.testing.assert_allclose(
            render.mesh_center(m), target, atol=1.0e-12
        )


def test_scene_arrange_line_spacing() -> None:
    """Verify gap spacing between adjacent mesh bounding boxes."""
    # Two unit boxes (width=1.0) with spacing=2.0 -> center-to-center = 3.0
    m1 = _make_unit_box((0, 0, 0))
    m2 = _make_unit_box((0, 0, 0))

    arranged = render.scene_arrange_line([m1, m2], axis=0, spacing=2.0)
    c1 = render.mesh_center(arranged[0])
    c2 = render.mesh_center(arranged[1])

    assert np.isclose(c2[0] - c1[0], 3.0)


def test_scene_arrange_grid_2x2() -> None:
    """Verify 2x2 grid layout and centering."""
    meshes = [_make_unit_box((0, 0, 0)) for _ in range(4)]
    arranged = render.scene_arrange_grid(
        meshes, columns=2, spacing=(1.0, 1.0), center=True
    )

    centers = [render.mesh_center(m) for m in arranged]
    # In a 2x2 grid of unit boxes with gap 1.0, centers should be at +/-1.0
    np.testing.assert_allclose(centers[0][:2], np.array([-1.0, -1.0]))
    np.testing.assert_allclose(centers[1][:2], np.array([1.0, -1.0]))
    np.testing.assert_allclose(centers[2][:2], np.array([-1.0, 1.0]))
    np.testing.assert_allclose(centers[3][:2], np.array([1.0, 1.0]))

    # Scene center must be at origin
    np.testing.assert_allclose(
        render.scene_center(arranged), np.zeros(3), atol=1.0e-12
    )


def test_scene_arrange_circle() -> None:
    """Verify 4 objects arranged on circle of radius R."""
    radius = 50.0
    meshes = [_make_unit_box((0, 0, 0)) for _ in range(4)]
    arranged = render.scene_arrange_circle(meshes, radius=radius, plane="xy")

    centers = [render.mesh_center(m) for m in arranged]
    # Angles: 0, 90, 180, 270 deg
    np.testing.assert_allclose(
        centers[0], np.array([radius, 0.0, 0.0]), atol=1.0e-12
    )
    np.testing.assert_allclose(
        centers[1], np.array([0.0, radius, 0.0]), atol=1.0e-12
    )
    np.testing.assert_allclose(
        centers[2], np.array([-radius, 0.0, 0.0]), atol=1.0e-12
    )
    np.testing.assert_allclose(
        centers[3], np.array([0.0, -radius, 0.0]), atol=1.0e-12
    )
