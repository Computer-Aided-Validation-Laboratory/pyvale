# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Verification tests for spatial and temporal integration sensors using exact
SymPy symbolic solutions on 2D and 3D finite element meshes.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import sympy

import pyvale.sensorsim as sens
import pyvale.verif as verif


def test_sensors_line_scalar_linear_2d() -> None:
    """Tests 1D line sensor measuring average linear scalar field on a
    2D mesh.
    """
    sim_data, data_gen = verif.scalar_linear_2d()

    # Place a horizontal line sensor of length 4.0 centered at (5.0, 3.75, 0.0)
    # The sensor extends from x=3.0 to x=7.0 at constant y=3.75
    length = 4.0
    center_x, center_y = 5.0, 3.75
    pos = np.array([[center_x, center_y, 0.0]])

    sens_data = sens.SensorData(positions=pos, sample_times=sim_data.time)
    field = sens.FieldScalar(
        sim_data=sim_data,
        comp_key="temperature",
        spatial_dims=sens.EDim.TWOD,
    )
    window = sens.SpatialWindowLine(
        length=length,
        axis=(1.0, 0.0, 0.0),
        integ_rule=sens.IntegrationGaussLegendre(order=2),
    )

    sensor_array = sens.SensorsSpatial(
        sensor_data=sens_data,
        field=field,
        spatial_window=window,
        integration_mode=sens.EIntegrationMode.AVERAGE,
    )

    meas = sensor_array.get_truth()  # shape (1, 1, n_times)

    # Exact symbolic average: (1/L) * integral f(x, y=3.75, t) dx
    for tt, t_val in enumerate(sim_data.time):
        exact_integral = data_gen.integrate_symbolic(
            field_key="temperature",
            bounds_x=(center_x - length / 2.0, center_x + length / 2.0),
            bounds_y=(center_y, center_y),
            bounds_t=(t_val, t_val),
        )
        exact_avg = exact_integral / length
        assert np.isclose(meas[0, 0, tt], exact_avg, rtol=1e-4)


def test_sensors_area_rectangle_scalar_quad_2d() -> None:
    """Tests 2D rectangular area sensor measuring quadratic field on
    a 2D mesh.
    """
    sim_data, data_gen = verif.scalar_quadratic_2d()

    # Rectangular sensor 2.0 x 2.0 centered at (5.0, 3.75, 0.0)
    # x in [4.0, 6.0], y in [2.75, 4.75]
    lx, ly = 2.0, 2.0
    center = np.array([[5.0, 3.75, 0.0]])
    sens_data = sens.SensorData(positions=center, sample_times=sim_data.time)

    field = sens.FieldScalar(
        sim_data=sim_data,
        comp_key="temperature",
        spatial_dims=sens.EDim.TWOD,
    )
    window = sens.SpatialWindowRectangle(
        length_x=lx,
        length_y=ly,
        integ_rule=sens.IntegrationGaussLegendre(order=3),
    )

    sensor_array = sens.SensorsSpatial(
        sensor_data=sens_data,
        field=field,
        spatial_window=window,
        integration_mode=sens.EIntegrationMode.AVERAGE,
    )

    meas = sensor_array.get_truth()

    area = lx * ly
    for tt, t_val in enumerate(sim_data.time):
        exact_integral = data_gen.integrate_symbolic(
            field_key="temperature",
            bounds_x=(5.0 - lx / 2.0, 5.0 + lx / 2.0),
            bounds_y=(3.75 - ly / 2.0, 3.75 + ly / 2.0),
            bounds_t=(t_val, t_val),
        )
        exact_avg = exact_integral / area
        assert np.isclose(meas[0, 0, tt], exact_avg, rtol=1e-3)


def test_sensors_volume_box_scalar_linear_3d() -> None:
    """Tests 3D box volume sensor measuring linear scalar field on a 3D mesh."""
    sim_data, data_gen = verif.scalar_linear_3d()

    # Box 2.0 x 2.0 x 1.0 centered at (5.0, 3.75, 2.5)
    lx, ly, lz = 2.0, 2.0, 1.0
    center = np.array([[5.0, 3.75, 2.5]])
    sens_data = sens.SensorData(positions=center, sample_times=sim_data.time)

    field = sens.FieldScalar(
        sim_data=sim_data,
        comp_key="temperature",
        spatial_dims=sens.EDim.THREED,
    )
    window = sens.SpatialWindowBox(
        length_x=lx,
        length_y=ly,
        length_z=lz,
        integ_rule=sens.IntegrationGaussLegendre(order=2),
    )

    sensor_avg = sens.SensorsSpatial(
        sensor_data=sens_data,
        field=field,
        spatial_window=window,
        integration_mode=sens.EIntegrationMode.AVERAGE,
    )

    sensor_acc = sens.SensorsSpatial(
        sensor_data=sens_data,
        field=field,
        spatial_window=window,
        integration_mode=sens.EIntegrationMode.ACCUMULATE,
    )

    meas_avg = sensor_avg.get_truth()
    meas_acc = sensor_acc.get_truth()

    vol = lx * ly * lz
    assert np.allclose(meas_acc, meas_avg * vol)

    for tt, t_val in enumerate(sim_data.time):
        exact_integral = data_gen.integrate_symbolic(
            field_key="temperature",
            bounds_x=(5.0 - lx / 2.0, 5.0 + lx / 2.0),
            bounds_y=(3.75 - ly / 2.0, 3.75 + ly / 2.0),
            bounds_z=(2.5 - lz / 2.0, 2.5 + lz / 2.0),
            bounds_t=(t_val, t_val),
        )
        assert np.isclose(meas_acc[0, 0, tt], exact_integral, rtol=1e-4)


def test_sensors_temporal_window_transient_3d() -> None:
    """Tests temporal integration window on dynamic 3D simulation data."""
    sim_data, data_gen = verif.scalar_linear_3d()

    # Time steps from 0.0 to 1.0
    center = np.array([[5.0, 3.75, 2.5]])
    eval_times = np.array([0.5])

    sens_data = sens.SensorData(positions=center, sample_times=eval_times)
    field = sens.FieldScalar(
        sim_data=sim_data,
        comp_key="temperature",
        spatial_dims=sens.EDim.THREED,
    )

    # Temporal window centered at t=0.5 with duration 0.2 -> [0.4, 0.6]
    temp_win = sens.TemporalWindowCentered(
        duration=0.2,
        integ_rule=sens.IntegrationGaussLegendre(order=3),
    )

    sensor = sens.SensorsSpatial(
        sensor_data=sens_data,
        field=field,
        spatial_window=sens.SpatialWindowPoint(),
        temporal_window=temp_win,
        integration_mode=sens.EIntegrationMode.AVERAGE,
    )

    meas = sensor.get_truth()

    # Exact temporal average over [0.4, 0.6]
    exact_integral = data_gen.integrate_symbolic(
        field_key="temperature",
        bounds_x=(5.0, 5.0),
        bounds_y=(3.75, 3.75),
        bounds_z=(2.5, 2.5),
        bounds_t=(0.4, 0.6),
    )
    exact_avg = exact_integral / 0.2
    assert np.isclose(meas[0, 0, 0], exact_avg, rtol=1e-4)
