# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import pytest
import numpy as np
import pyvale.verif as verif
from pyvale.sensorsim.enums import EDim, EDifferentialMode, EIntegrationMode
from pyvale.sensorsim.sensordata import SensorData
from pyvale.sensorsim.fieldscalar import FieldScalar
from pyvale.sensorsim.fieldvector import FieldVector
from pyvale.sensorsim.sensorsspatial import SensorsSpatial
from pyvale.sensorsim.sensorsdifferential import SensorsDifferential
from pyvale.sensorsim.spatialwindows import (
    SpatialWindowPoint,
    SpatialWindowLine,
)
from pyvale.sensorsim.errorsysindep import ErrSysOffset
from pyvale.sensorsim.errorrand import ErrRandGen
from pyvale.sensorsim.generatorsrandom import GenNormal


def test_differential_temperature_difference() -> None:
    sim_data, data_gen = verif.scalar_linear_2d()

    # Anchor A at (3.0, 3.0, 0.0), Anchor B at (7.0, 3.0, 0.0)
    field_a = FieldScalar(sim_data, "temperature", EDim.TWOD)
    field_b = FieldScalar(sim_data, "temperature", EDim.TWOD)

    sens_data_a = SensorData(positions=np.array([[3.0, 3.0, 0.0]]))
    sens_data_b = SensorData(positions=np.array([[7.0, 3.0, 0.0]]))

    sens_a = SensorsSpatial(sens_data_a, field_a)
    sens_b = SensorsSpatial(sens_data_b, field_b)

    diff_sensor = SensorsDifferential(
        sensor_a=sens_a,
        sensor_b=sens_b,
        mode=EDifferentialMode.DIFFERENCE,
    )

    truth = diff_sensor.get_truth()
    # Linear field in x has slope 20.0 / 10.0 = 2.0 per unit x
    # dx = 7.0 - 3.0 = 4.0 -> delta T = 4.0 * 2.0 = 8.0 (at t=0, scaled by time)
    # Check shape
    assert truth.shape == (1, 1, sim_data.time.shape[0])
    truth_a = sens_a.get_truth()
    truth_b = sens_b.get_truth()
    assert np.allclose(truth, truth_b - truth_a)


def test_extensometer_strain_mode() -> None:
    sim_data, data_gen = verif.scalar_linear_2d()

    n_times = sim_data.time.shape[0]
    ux = 0.002 * sim_data.coords[:, 0:1]
    sim_data.node_vars["disp_x"] = np.tile(ux, (1, n_times))
    sim_data.node_vars["disp_y"] = np.zeros(
        (sim_data.coords.shape[0], n_times), dtype=np.float64
    )

    field_a = FieldVector(sim_data, ("disp_x", "disp_y"), EDim.TWOD)
    field_b = FieldVector(sim_data, ("disp_x", "disp_y"), EDim.TWOD)

    # Gauge length = 4.0 mm from x=3.0 to x=7.0
    sens_data_a = SensorData(positions=np.array([[3.0, 3.0, 0.0]]))
    sens_data_b = SensorData(positions=np.array([[7.0, 3.0, 0.0]]))

    s_win = SpatialWindowLine(length=2.0, axis=(0.0, 1.0, 0.0))

    sens_a = SensorsSpatial(sens_data_a, field_a, spatial_window=s_win)
    sens_b = SensorsSpatial(sens_data_b, field_b, spatial_window=s_win)

    ext = SensorsDifferential(
        sensor_a=sens_a,
        sensor_b=sens_b,
        mode=EDifferentialMode.STRAIN,
    )

    truth = ext.get_truth()
    # eps = (u_xB - u_xA) / L0 = (0.002*7 - 0.002*3) / 4.0 = 0.0020
    assert np.isclose(truth[0, 0, 0], 0.0020, atol=1e-5)


def test_differential_errors_and_custom_func() -> None:
    sim_data, data_gen = verif.scalar_linear_2d()

    field = FieldScalar(sim_data, "temperature", EDim.TWOD)
    sens_data_a = SensorData(positions=np.array([[3.0, 3.0, 0.0]]))
    sens_data_b = SensorData(positions=np.array([[7.0, 3.0, 0.0]]))

    sens_a = SensorsSpatial(sens_data_a, field)
    sens_b = SensorsSpatial(sens_data_b, field)

    custom_fn = lambda a, b: (b - a) / 2.0
    diff_sensor = SensorsDifferential(
        sensor_a=sens_a,
        sensor_b=sens_b,
        mode=EDifferentialMode.CUSTOM,
        custom_func=custom_fn,
    )

    truth = diff_sensor.get_truth()
    truth_a = sens_a.get_truth()
    truth_b = sens_b.get_truth()
    expected = (truth_b - truth_a) / 2.0
    assert np.allclose(truth, expected)

    err_chain = (
        ErrSysOffset(offset=0.5),
        ErrRandGen(GenNormal(std=0.1, mean=0.0)),
    )
    diff_sensor.set_error_chain(err_chain)

    meas = diff_sensor.sim_measurements()
    err_tot = diff_sensor.get_errors_total()
    assert err_tot is not None
    assert np.allclose(meas, truth + err_tot)
