# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Unit tests for spatial windows, 3D rotations, and coordinate transforms.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pyvale.sensorsim.enums import EIntegrationMode
from pyvale.sensorsim.spatialwindows import (
    SpatialWindowPoint,
    SpatialWindowLine,
    SpatialWindowRectangle,
    SpatialWindowDisk,
    SpatialWindowBox,
    SpatialWindowCylinder,
    SpatialWindowSphere,
)
from pyvale.sensorsim.sensortools import (
    orient_from_direction,
    orient_from_normal_and_tangent,
)


def test_spatial_window_line() -> None:
    """Test 1D line sensor window measure and rotated global points."""
    length = 10.0
    line = SpatialWindowLine(length=length)
    assert line.get_spatial_dims() == 1
    assert np.isclose(line.get_measure(), length)

    pts, w_avg = line.get_local_points_and_weights(
        mode=EIntegrationMode.AVERAGE
    )
    assert np.isclose(np.sum(w_avg), 1.0)
    assert pts.shape[1] == 3
    assert np.allclose(pts[:, 1:], 0.0)  # along local X axis
    assert np.all(pts[:, 0] >= -5.0) and np.all(pts[:, 0] <= 5.0)

    # Rotate by 90 degrees around Y axis (maps X to -Z)
    rot_y = Rotation.from_euler("y", 90.0, degrees=True)
    sensor_pos = np.array([[10.0, 20.0, 30.0]])
    glob_pts = line.to_global_points(sensor_pos, (rot_y,))

    assert glob_pts.shape == (1, pts.shape[0], 3)
    # Check that points vary along Z and are constant in X and Y
    assert np.allclose(glob_pts[0, :, 0], 10.0)
    assert np.allclose(glob_pts[0, :, 1], 20.0)
    assert np.ptp(glob_pts[0, :, 2]) > 0.0


def test_spatial_window_rectangle_and_disk() -> None:
    """Test 2D rectangle and disk measures."""
    rect = SpatialWindowRectangle(length_x=4.0, length_y=3.0)
    assert rect.get_spatial_dims() == 2
    assert np.isclose(rect.get_measure(), 12.0)
    _, w_rect_acc = rect.get_local_points_and_weights(
        mode=EIntegrationMode.ACCUMULATE
    )
    assert np.isclose(np.sum(w_rect_acc), 12.0)

    disk = SpatialWindowDisk(radius=2.5)
    assert disk.get_spatial_dims() == 2
    assert np.isclose(disk.get_measure(), np.pi * 2.5**2)
    _, w_disk_acc = disk.get_local_points_and_weights(
        mode=EIntegrationMode.ACCUMULATE
    )
    assert np.isclose(np.sum(w_disk_acc), np.pi * 2.5**2, rtol=1e-3)


def test_spatial_window_3d_volumes() -> None:
    """Test 3D box, cylinder, and sphere volume integrations."""
    box = SpatialWindowBox(length_x=2.0, length_y=3.0, length_z=4.0)
    assert box.get_spatial_dims() == 3
    assert np.isclose(box.get_measure(), 24.0)

    cyl = SpatialWindowCylinder(radius=2.0, height=5.0)
    assert cyl.get_spatial_dims() == 3
    assert np.isclose(cyl.get_measure(), np.pi * 4.0 * 5.0)
    _, w_cyl = cyl.get_local_points_and_weights(
        mode=EIntegrationMode.ACCUMULATE
    )
    assert np.isclose(np.sum(w_cyl), np.pi * 4.0 * 5.0, rtol=1e-3)

    sph = SpatialWindowSphere(radius=3.0)
    assert sph.get_spatial_dims() == 3
    expected_vol = (4.0 / 3.0) * np.pi * 27.0
    assert np.isclose(sph.get_measure(), expected_vol)
    _, w_sph = sph.get_local_points_and_weights(
        mode=EIntegrationMode.ACCUMULATE
    )
    assert np.isclose(np.sum(w_sph), expected_vol, rtol=1e-2)


def test_orient_from_direction() -> None:
    """Test orient_from_direction aligning e1 with arbitrary target vectors."""
    targets = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [-2.0, 3.5, 1.2],
    ]
    e1 = np.array([1.0, 0.0, 0.0])

    for t in targets:
        rot = orient_from_direction(t)
        transformed = rot.apply(e1)
        expected = np.array(t) / np.linalg.norm(t)
        np.testing.assert_allclose(transformed, expected, atol=1e-6)


def test_orient_from_normal_and_tangent() -> None:
    """Test orient_from_normal_and_tangent constructing orthogonal frames."""
    normal = (0.0, 0.0, 1.0)
    tangent = (1.0, 1.0, 0.0)

    rot = orient_from_normal_and_tangent(normal, tangent)
    matrix = rot.as_matrix()

    # Column 0 is e1 (tangent), Column 1 is e2, Column 2 is e3 (normal)
    e1 = matrix[:, 0]
    e2 = matrix[:, 1]
    e3 = matrix[:, 2]

    # Orthogonality
    assert np.isclose(np.dot(e1, e2), 0.0, atol=1e-7)
    assert np.isclose(np.dot(e2, e3), 0.0, atol=1e-7)
    assert np.isclose(np.dot(e1, e3), 0.0, atol=1e-7)

    # e3 aligns with normal
    assert np.allclose(e3, [0.0, 0.0, 1.0])

    # e1 aligns with normalized tangent
    assert np.allclose(e1, np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0))


def test_tensor_rotation_transformation() -> None:
    """Test 3D strain tensor rotation under 45 degree yaw."""
    # Global strain tensor with sigma_xx=100, sigma_yy=50
    sigma_global = np.array([
        [100.0, 0.0, 0.0],
        [0.0, 50.0, 0.0],
        [0.0, 0.0, 0.0],
    ])

    # Rotate sensor by 45 degrees around Z: R^T * sigma * R
    rot = Rotation.from_euler("z", 45.0, degrees=True)
    R = rot.as_matrix()
    sigma_local = R.T @ sigma_global @ R

    # Expected: (100+50)/2 = 75 for normal strains, (100-50)/2 = 25 for shear
    assert np.isclose(sigma_local[0, 0], 75.0)
    assert np.isclose(sigma_local[1, 1], 75.0)
    assert np.isclose(sigma_local[0, 1], -25.0)
    assert np.isclose(sigma_local[1, 0], -25.0)
