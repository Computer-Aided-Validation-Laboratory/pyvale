# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Analytic tests for camera placement, orientation, projection, and framing."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import pyvale.render as render


def _make_test_camera(
    pos=(0.0, 0.0, 2.0),
    pixels_num=(512, 512),
    pixels_size=(0.02, 0.02),
    focal_length=1.0,
) -> render.Camera:
    return render.Camera(
        pixels_num=np.array(pixels_num),
        pixels_size=np.array(pixels_size),
        pos_world=np.array(pos, dtype=np.float64),
        rot_world=Rotation.identity(),
        roi_cent_world=np.zeros(3),
        focal_length=focal_length,
    )


def test_cam_look_at_optical_axis_alignment() -> None:
    """Verify camera basis points towards target and basis is orthonormal."""
    cam = _make_test_camera(pos=(0.0, 0.0, 10.0))
    oriented = render.cam_look_at(cam, target=(0.0, 0.0, 0.0))

    # Look along -Z: rot_world should be identity
    rot_mat = oriented.rot_world.as_matrix()
    np.testing.assert_allclose(rot_mat, np.eye(3), atol=1.0e-12)
    assert np.isclose(np.linalg.det(rot_mat), 1.0)


def test_cam_look_at_arbitrary_target() -> None:
    """Look from (1, 2, 3) to (4, 6, 3) with view vector (3, 4, 0)."""
    cam = _make_test_camera(pos=(1.0, 2.0, 3.0))
    oriented = render.cam_look_at(cam, target=(4.0, 6.0, 3.0))

    rot_mat = oriented.rot_world.as_matrix()
    # Camera forward is -Z in camera coords, which must equal (3, 4, 0) / 5
    z_cam = rot_mat[:, 2]
    expected_forward = np.array([3.0, 4.0, 0.0]) / 5.0
    np.testing.assert_allclose(-z_cam, expected_forward, atol=1.0e-12)


def test_cam_look_at_degeneracy_raises() -> None:
    """Coincident camera position and target raises ValueError."""
    cam = _make_test_camera(pos=(1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="coincident"):
        render.cam_look_at(cam, target=(1.0, 2.0, 3.0))


def test_cam_project_points_optical_axis_centers() -> None:
    """A point on optical axis projects to image center (cx, cy)."""
    cam = _make_test_camera(
        pos=(0.0, 0.0, 2.0),
        pixels_num=(512, 512),
        pixels_size=(0.02, 0.02),
        focal_length=1.0,
    )
    # Point at (0, 0, 0) lies on optical axis at depth 2.0
    px = render.cam_project_points(cam, np.array([[0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(px[0], np.array([256.0, 256.0]), atol=1.0e-12)


def test_cam_project_points_perspective_depth_scaling() -> None:
    """Projected offset from center scales inversely with depth (1/z)."""
    cam = _make_test_camera(
        pos=(0.0, 0.0, 10.0),
        pixels_num=(512, 512),
        pixels_size=(0.02, 0.02),
        focal_length=1.0,
    )
    # Point 1 at depth = 2.0 (z=8.0), offset x=1.0
    p1 = np.array([[1.0, 0.0, 8.0]])
    # Point 2 at depth = 4.0 (z=6.0), offset x=1.0
    p2 = np.array([[1.0, 0.0, 6.0]])

    px1 = render.cam_project_points(cam, p1)[0]
    px2 = render.cam_project_points(cam, p2)[0]

    offset1 = px1[0] - 256.0
    offset2 = px2[0] - 256.0

    # offset1 / offset2 should be depth2 / depth1 = 4.0 / 2.0 = 2.0
    assert np.isclose(offset1 / offset2, 2.0)


def test_cam_frame_mesh_and_scene() -> None:
    """Framing a mesh and framing a scene produce valid camera positions."""
    coords = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    mesh = render.Mesh3D(
        element_type=render.EElementType.TRI3,
        coords=coords,
        connectivity=np.array([[0, 1, 2]], dtype=np.uintp),
        shader=None,
    )
    cam = _make_test_camera(pos=(0.0, 0.0, 2.0))

    framed_mesh = render.cam_frame_mesh(cam, mesh, fill=0.9)
    assert framed_mesh.pos_world[2] > 0.0

    framed_scene = render.cam_frame_scene(cam, [mesh], fill=0.9)
    np.testing.assert_allclose(framed_scene.pos_world, framed_mesh.pos_world)


def test_cam_stereo_symmetric_and_faceon() -> None:
    """Verify stereo baseline and symmetric / face-on angles."""
    cam = _make_test_camera(pos=(0.0, 0.0, 10.0))
    angle = 20.0

    stereo_sym = render.cam_stereo_symmetric(cam, angle)
    half_angle_rad = np.radians(angle / 2.0)
    expected_baseline = 2.0 * 10.0 * np.tan(half_angle_rad)
    actual_baseline = (
        stereo_sym.camera_1.pos_world[0] - stereo_sym.camera_0.pos_world[0]
    )
    assert np.isclose(actual_baseline, expected_baseline)

    stereo_faceon = render.cam_stereo_faceon(cam, angle)
    expected_faceon_base = 10.0 * np.tan(np.radians(angle))
    actual_faceon_base = (
        stereo_faceon.camera_1.pos_world[0]
        - stereo_faceon.camera_0.pos_world[0]
    )
    assert np.isclose(actual_faceon_base, expected_faceon_base)
