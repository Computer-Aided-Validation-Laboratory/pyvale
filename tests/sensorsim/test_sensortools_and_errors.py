# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Tests for sensor placement helpers, 3D element types, 2D rotation guards,
and calibration inversion.
"""

import numpy as np
import pytest
import pyvista as pv
from scipy.spatial.transform import Rotation

from pyvale.sensorsim.enums import EDim
from pyvale.sensorsim.fieldconverter import _get_pyvista_cell_type
from pyvale.sensorsim.fieldtransform import validate_rotation_planar_2d
from pyvale.sensorsim.sensordata import SensorData
from pyvale.sensorsim.sensortools import (
    gen_pos_grid_inside,
    gen_pos_grid_boundary,
    gen_pos_cylinder,
    gen_pos_sphere,
)
from pyvale.sensorsim.errorsyscalib import ErrSysCalibration
from pyvale.sensorsim.errorsysfield import _perturb_sample_times


def test_pyvista_cell_types_3d_wedges_pyramids() -> None:
    """Test 3D wedge and pyramid element mappings."""
    assert _get_pyvista_cell_type(6, EDim.THREED) == pv.CellType.WEDGE
    assert (
        _get_pyvista_cell_type(15, EDim.THREED)
        == pv.CellType.QUADRATIC_WEDGE
    )
    assert _get_pyvista_cell_type(5, EDim.THREED) == pv.CellType.PYRAMID
    assert (
        _get_pyvista_cell_type(13, EDim.THREED)
        == pv.CellType.QUADRATIC_PYRAMID
    )
    assert _get_pyvista_cell_type(8, EDim.THREED) == pv.CellType.HEXAHEDRON
    assert _get_pyvista_cell_type(4, EDim.THREED) == pv.CellType.TETRA


def test_validate_rotation_planar_2d() -> None:
    """Test 2D planar rotation validation."""
    # Planar Z rotation should pass
    rot_z = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    validate_rotation_planar_2d(rot_z)

    # Out-of-plane X or Y rotation should raise ValueError
    rot_x = Rotation.from_euler("x", 30, degrees=True).as_matrix()
    with pytest.raises(ValueError, match="Out-of-plane rotation"):
        validate_rotation_planar_2d(rot_x)

    rot_y = Rotation.from_euler("y", 15, degrees=True).as_matrix()
    with pytest.raises(ValueError, match="Out-of-plane rotation"):
        validate_rotation_planar_2d(rot_y)


def test_gen_pos_grid_boundary() -> None:
    """Test boundary-inclusive grid placement."""
    grid = gen_pos_grid_boundary(
        num_sensors=(3, 3, 2),
        x_lims=(0.0, 10.0),
        y_lims=(0.0, 20.0),
        z_lims=(0.0, 5.0),
    )
    assert grid.shape == (3 * 3 * 2, 3)
    assert np.min(grid[:, 0]) == pytest.approx(0.0)
    assert np.max(grid[:, 0]) == pytest.approx(10.0)
    assert np.min(grid[:, 1]) == pytest.approx(0.0)
    assert np.max(grid[:, 1]) == pytest.approx(20.0)
    assert np.min(grid[:, 2]) == pytest.approx(0.0)
    assert np.max(grid[:, 2]) == pytest.approx(5.0)


def test_gen_pos_cylinder() -> None:
    """Test cylindrical surface placement."""
    radius = 5.0
    cyl = gen_pos_cylinder(
        num_theta=8,
        num_z=4,
        radius=radius,
        z_lims=(0.0, 10.0),
        center=(1.0, 2.0, 0.0),
    )
    assert cyl.shape == (8 * 4, 3)
    # Check radii from center (1, 2)
    dist_xy = np.sqrt((cyl[:, 0] - 1.0) ** 2 + (cyl[:, 1] - 2.0) ** 2)
    assert np.allclose(dist_xy, radius)
    assert np.min(cyl[:, 2]) == pytest.approx(0.0)
    assert np.max(cyl[:, 2]) == pytest.approx(10.0)


def test_gen_pos_sphere() -> None:
    """Test spherical surface placement."""
    radius = 10.0
    num_sensors = 50
    center = (2.0, -3.0, 4.0)
    sphere = gen_pos_sphere(
        num_sensors=num_sensors,
        radius=radius,
        center=center,
    )
    assert sphere.shape == (num_sensors, 3)
    # Check radii from center
    dist_r = np.sqrt(
        (sphere[:, 0] - center[0]) ** 2
        + (sphere[:, 1] - center[1]) ** 2
        + (sphere[:, 2] - center[2]) ** 2
    )
    assert np.allclose(dist_r, radius)


def test_calibration_inversion_newton() -> None:
    """Test Newton-Raphson calibration inversion against non-linear function."""
    # Quadratic calibration: truth V = 2 * T + 0.1 * T^2
    def truth_calib(v: np.ndarray) -> np.ndarray:
        return 2.0 * v + 0.1 * v**2

    def truth_prime(v: np.ndarray) -> np.ndarray:
        return 2.0 + 0.2 * v

    def assumed_calib(v: np.ndarray) -> np.ndarray:
        return 2.0 * v  # Linear assumption ignoring quadratic term

    cal_sim = ErrSysCalibration(
        assumed_calib=assumed_calib,
        truth_calib=truth_calib,
        truth_calib_prime=truth_prime,
        cal_range=(0.0, 10.0),
        use_newton=True,
    )

    dummy_sens = SensorData()
    err_basis = np.array([[[10.0, 20.0, 30.0]]], dtype=np.float64)
    errs, _ = cal_sim.sim_errs(err_basis, dummy_sens)

    assert errs.shape == err_basis.shape
    # For basis 10.0: 2*V + 0.1*V^2 = 10 -> exact root is V = (-2 + sqrt(8))/0.2
    v_exact = (-2.0 + np.sqrt(4.0 + 4.0)) / 0.2
    expected_err = assumed_calib(np.array([v_exact]))[0] - 10.0
    assert errs[0, 0, 0] == pytest.approx(expected_err, abs=1e-6)


def test_perturb_sample_times_per_sensor() -> None:
    """Test 2D per-sensor time perturbations."""
    sim_time = np.linspace(0.0, 1.0, 11)
    # 2D time offset for 3 sensors
    time_offset_2d = np.array([
        [0.01] * 11,
        [0.02] * 11,
        [0.03] * 11,
    ])

    perturbed = _perturb_sample_times(
        sim_time=sim_time,
        time_nominal=None,
        time_offset=time_offset_2d,
        time_rand=None,
        time_drift=None,
    )

    assert perturbed.shape == (3, 11)
    assert perturbed[0, 0] == pytest.approx(0.01)
    assert perturbed[1, 0] == pytest.approx(0.02)
    assert perturbed[2, 0] == pytest.approx(0.03)
