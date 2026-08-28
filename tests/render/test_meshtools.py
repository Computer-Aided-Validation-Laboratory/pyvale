# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Analytic tests for mesh transformation and frame helpers."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import pyvale.render as render


def _make_sample_mesh3d() -> render.Mesh3D:
    coords = np.array([
        [-2.0, -3.0, -4.0],
        [5.0, 7.0, 11.0],
        [1.0, 2.0, 3.0],
    ])
    connect = np.array([[0, 1, 2]], dtype=np.uintp)
    return render.Mesh3D(
        element_type=render.EElementType.TRI3,
        coords=coords,
        connectivity=connect,
        shader=None,
    )


def test_mesh_bounds_and_center_analytic() -> None:
    """Verify exact min/max bounds and bounding box center."""
    mesh = _make_sample_mesh3d()
    lower, upper = render.mesh_bounds(mesh)
    np.testing.assert_allclose(lower, np.array([-2.0, -3.0, -4.0]))
    np.testing.assert_allclose(upper, np.array([5.0, 7.0, 11.0]))

    center = render.mesh_center(mesh)
    np.testing.assert_allclose(center, np.array([1.5, 2.0, 3.5]))


def test_mesh_translate_preserves_purity_and_vectors() -> None:
    """Translation shifts coordinates without modifying displacements."""
    mesh = _make_sample_mesh3d()
    mesh.displacements = np.ones((1, 3, 3))

    offset = [10.0, -2.0, 5.0]
    moved = render.mesh_translate(mesh, offset)

    expected_coords = mesh.coords + np.array(offset)
    np.testing.assert_allclose(moved.coords, expected_coords)
    np.testing.assert_allclose(moved.displacements, mesh.displacements)
    # Check original was not mutated
    assert not np.array_equal(mesh.coords, moved.coords)


def test_mesh_rotate_z_axis_90_degrees() -> None:
    """Rotate (1, 0, 0) by 90 deg about Z -> (0, 1, 0)."""
    coords = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mesh = render.Mesh3D(
        element_type=render.EElementType.TRI3,
        coords=coords,
        connectivity=np.array([[0, 1, 0]], dtype=np.uintp),
        shader=None,
    )
    rot = Rotation.from_euler("z", 90.0, degrees=True)
    rotated = render.mesh_rotate(mesh, rot)

    expected = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    np.testing.assert_allclose(rotated.coords, expected, atol=1.0e-12)


def test_mesh_rotate_with_pivot() -> None:
    """Rotate (2, 0, 0) around pivot (1, 0, 0) by 180 deg -> (0, 0, 0)."""
    coords = np.array([[2.0, 0.0, 0.0]])
    mesh = render.Mesh3D(
        element_type=render.EElementType.TRI3,
        coords=coords,
        connectivity=np.array([[0, 0, 0]], dtype=np.uintp),
        shader=None,
    )
    rot = Rotation.from_euler("z", 180.0, degrees=True)
    rotated = render.mesh_rotate(mesh, rot, pivot=[1.0, 0.0, 0.0])

    expected = np.array([[0.0, 0.0, 0.0]])
    np.testing.assert_allclose(rotated.coords, expected, atol=1.0e-12)


def test_mesh_rotate_rotates_displacement_vectors() -> None:
    """Displacement vector (1, 0, 0) rotated by 90 deg about Z -> (0, 1, 0)."""
    mesh = render.Mesh3D(
        element_type=render.EElementType.TRI3,
        coords=np.array([[0.0, 0.0, 0.0]]),
        connectivity=np.array([[0, 0, 0]], dtype=np.uintp),
        shader=None,
        displacements=np.array([[[1.0, 0.0, 0.0]]]),
    )
    rot = Rotation.from_euler("z", 90.0, degrees=True)
    rotated = render.mesh_rotate(mesh, rot)

    expected_disp = np.array([[[0.0, 1.0, 0.0]]])
    np.testing.assert_allclose(
        rotated.displacements, expected_disp, atol=1.0e-12
    )


def test_mesh_scale_uniform_and_axes() -> None:
    """Scale coordinates uniformly and along axes."""
    coords = np.array([[1.0, 2.0, 3.0]])
    mesh = render.Mesh3D(
        element_type=render.EElementType.TRI3,
        coords=coords,
        connectivity=np.array([[0, 0, 0]], dtype=np.uintp),
        shader=None,
    )
    scaled_uniform = render.mesh_scale(mesh, 2.0)
    np.testing.assert_allclose(
        scaled_uniform.coords, np.array([[2.0, 4.0, 6.0]])
    )

    scaled_axes = render.mesh_scale(mesh, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(scaled_axes.coords, np.array([[2.0, 6.0, 12.0]]))


def test_mesh_center_at() -> None:
    """Centering a mesh translates its center to the target."""
    mesh = _make_sample_mesh3d()
    centered = render.mesh_center_at(mesh, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        render.mesh_center(centered), np.zeros(3), atol=1.0e-12
    )

    offset_target = [10.0, 20.0, 30.0]
    placed = render.mesh_center_at(mesh, offset_target)
    np.testing.assert_allclose(
        render.mesh_center(placed), offset_target, atol=1.0e-12
    )


def test_mesh_transform_composite() -> None:
    """Scale, rotate, and translate in canonical order."""
    coords = np.array([[1.0, 0.0, 0.0]])
    mesh = render.Mesh3D(
        element_type=render.EElementType.TRI3,
        coords=coords,
        connectivity=np.array([[0, 0, 0]], dtype=np.uintp),
        shader=None,
    )
    # 1. Scale by 2 -> (2, 0, 0)
    # 2. Rotate 90 deg about Z -> (0, 2, 0)
    # 3. Translate by (10, 0, 0) -> (10, 2, 0)
    transformed = render.mesh_transform(
        mesh,
        translation=[10.0, 0.0, 0.0],
        rotation=Rotation.from_euler("z", 90.0, degrees=True),
        scale=2.0,
    )
    expected = np.array([[10.0, 2.0, 0.0]])
    np.testing.assert_allclose(transformed.coords, expected, atol=1.0e-12)


def test_frame_index_helpers() -> None:
    """Verify deterministic frame sampling and selection."""
    indices = render.evenly_spaced_frame_indices(10, 4)
    np.testing.assert_array_equal(indices, np.array([0, 3, 6, 9]))

    fl_indices = render.first_last_frame_indices(10)
    np.testing.assert_array_equal(fl_indices, np.array([0, 9]))

    frames = np.arange(50).reshape(10, 5)
    selected = render.select_frames(frames, [0, 9])
    assert selected.shape == (2, 5)
    np.testing.assert_array_equal(selected[0], frames[0])
    np.testing.assert_array_equal(selected[1], frames[9])
