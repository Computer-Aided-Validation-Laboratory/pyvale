# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import pytest
import numpy as np
import pyvale.verif as verif
from pyvale.dataio.simdata import SimData
from pyvale.sensorsim.enums import EDim
from pyvale.sensorsim.sensorlibrary import SensorLibrary


@pytest.fixture
def mock_sim_data() -> SimData:
    sim_data, data_gen = verif.scalar_linear_2d()
    n_pts = sim_data.coords.shape[0]
    n_times = sim_data.time.shape[0]

    # Populate mechanics, temperature, flux, pressure
    ux = 0.001 * sim_data.coords[:, 0:1]
    uy = -0.0003 * sim_data.coords[:, 1:2]
    sim_data.node_vars["disp_x"] = np.tile(ux, (1, n_times))
    sim_data.node_vars["disp_y"] = np.tile(uy, (1, n_times))
    sim_data.node_vars["disp_z"] = np.zeros((n_pts, n_times))

    sim_data.node_vars["strain_xx"] = 1000e-6 * np.ones((n_pts, n_times))
    sim_data.node_vars["strain_yy"] = -300e-6 * np.ones((n_pts, n_times))
    sim_data.node_vars["strain_xy"] = np.zeros((n_pts, n_times))

    sim_data.node_vars["sigma_xx"] = 200.0 * np.ones((n_pts, n_times))
    sim_data.node_vars["sigma_yy"] = 50.0 * np.ones((n_pts, n_times))
    sim_data.node_vars["sigma_xy"] = 20.0 * np.ones((n_pts, n_times))

    sim_data.node_vars["flux_x"] = 100.0 * np.ones((n_pts, n_times))
    sim_data.node_vars["flux_y"] = 0.0 * np.ones((n_pts, n_times))
    sim_data.node_vars["flux_z"] = 500.0 * np.ones((n_pts, n_times))

    sim_data.node_vars["vel_x"] = 2.5 * np.ones((n_pts, n_times))
    sim_data.node_vars["vel_y"] = 0.0 * np.ones((n_pts, n_times))
    sim_data.node_vars["vel_z"] = 0.0 * np.ones((n_pts, n_times))

    sim_data.node_vars["pressure"] = 5.0 * np.ones((n_pts, n_times))
    return sim_data


def test_library_thermocouple(mock_sim_data: SimData) -> None:
    # Ideal mode
    tc_ideal = SensorLibrary.thermocouple(
        mock_sim_data,
        positions=np.array([[5.0, 3.75, 0.0]]),
        with_meas_errs=False,
    )
    truth = tc_ideal.get_truth()
    meas = tc_ideal.sim_measurements()
    assert np.allclose(truth, meas)
    assert tc_ideal.get_errors_total() is None

    # With errors
    tc_err = SensorLibrary.thermocouple(
        mock_sim_data,
        positions=np.array([[5.0, 3.75, 0.0]]),
        with_meas_errs=True,
    )
    meas_err = tc_err.sim_measurements()
    assert tc_err.get_errors_total() is not None
    assert not np.allclose(tc_err.get_truth(), meas_err)


def test_library_strain_gauge_and_rosette(mock_sim_data: SimData) -> None:
    sg = SensorLibrary.strain_gauge(
        mock_sim_data,
        positions=np.array([[5.0, 3.75, 0.0]]),
        with_meas_errs=False,
    )
    truth_sg = sg.get_truth()
    meas_sg = sg.sim_measurements()
    assert np.allclose(truth_sg, meas_sg)

    rosette = SensorLibrary.strain_rosette(
        mock_sim_data,
        position=(5.0, 3.75, 0.0),
        angles_deg=(0.0, 45.0, 90.0),
        with_meas_errs=False,
    )
    truth_ros = rosette.get_truth()
    assert truth_ros.shape[0] == 3  # 3 branches


def test_library_fbg_fiber(mock_sim_data: SimData) -> None:
    fiber = SensorLibrary.fbg_fiber(
        mock_sim_data,
        point_start=(2.0, 3.75, 0.0),
        point_end=(8.0, 3.75, 0.0),
        with_meas_errs=False,
    )
    truth = fiber.get_truth()
    assert truth.shape == (1, 1, mock_sim_data.time.shape[0])


def test_library_extensometer_and_lvdt(mock_sim_data: SimData) -> None:
    ext = SensorLibrary.extensometer(
        mock_sim_data,
        anchor_a=(2.0, 3.75, 0.0),
        anchor_b=(8.0, 3.75, 0.0),
        with_meas_errs=False,
    )
    truth_ext = ext.get_truth()
    # eps = (0.001*8 - 0.001*2) / 6.0 = 0.0010
    assert np.isclose(truth_ext[0, 0, 0], 0.0010, atol=1e-5)

    lvdt = SensorLibrary.lvdt(
        mock_sim_data,
        target_position=(5.0, 3.75, 0.0),
        axis=(1.0, 0.0, 0.0),
        spatial_dims=EDim.TWOD,
        with_meas_errs=False,
    )
    truth_lvdt = lvdt.get_truth()
    # u_x at x=5.0 is 0.001 * 5.0 = 0.0050 mm
    assert np.isclose(truth_lvdt[0, 0, 0], 0.0050, atol=1e-5)


def test_library_load_cell_and_flux() -> None:
    sim_data, data_gen = verif.scalar_quadratic_3d()
    n_pts = sim_data.coords.shape[0]
    n_times = sim_data.time.shape[0]

    sim_data.node_vars["sigma_xx"] = 200.0 * np.ones((n_pts, n_times))
    sim_data.node_vars["sigma_yy"] = 50.0 * np.ones((n_pts, n_times))
    sim_data.node_vars["sigma_zz"] = 100.0 * np.ones((n_pts, n_times))
    sim_data.node_vars["sigma_yz"] = np.zeros((n_pts, n_times))
    sim_data.node_vars["sigma_xz"] = np.zeros((n_pts, n_times))
    sim_data.node_vars["sigma_xy"] = np.zeros((n_pts, n_times))

    sim_data.node_vars["flux_x"] = 100.0 * np.ones((n_pts, n_times))
    sim_data.node_vars["flux_y"] = 0.0 * np.ones((n_pts, n_times))
    sim_data.node_vars["flux_z"] = 500.0 * np.ones((n_pts, n_times))

    lc = SensorLibrary.load_cell(
        sim_data,
        mount_position=(5.0, 3.75, 2.5),
        contact_area_x=2.0,
        contact_area_y=2.0,
        normal=(0.0, 0.0, 1.0),
        spatial_dims=EDim.THREED,
        with_meas_errs=False,
    )
    truth_lc = lc.get_truth()
    # Contact area = 4.0, normal is z -> Fz = 4.0 * 100.0 = 400.0 N
    assert np.isclose(truth_lc[0, 2, 0], 400.0, atol=1e-3)

    flux = SensorLibrary.heat_flux_meter(
        sim_data,
        position=(5.0, 3.75, 2.5),
        foil_radius=1.0,
        normal=(0.0, 0.0, 1.0),
        flux_keys=("flux_x", "flux_y", "flux_z"),
        spatial_dims=EDim.THREED,
        with_meas_errs=False,
    )
    truth_flux = flux.get_truth()
    # Normal flux: q_z = 500.0 W/m^2
    assert np.isclose(truth_flux[0, 0, 0], 500.0, atol=1e-3)


def test_library_ray_sensors(mock_sim_data: SimData) -> None:
    lidar = SensorLibrary.lidar(
        mock_sim_data,
        scanner_position=(5.0, 3.75, 50.0),
        beam_direction=(0.0, 0.0, -1.0),
        with_meas_errs=False,
    )
    truth_lidar = lidar.get_truth()
    assert np.allclose(truth_lidar[0, 0, :], 50.0, atol=1e-3)

    pyro = SensorLibrary.pyrometer(
        mock_sim_data,
        sensor_position=(5.0, 3.75, 50.0),
        aim_direction=(0.0, 0.0, -1.0),
        with_meas_errs=False,
    )
    truth_pyro = pyro.get_truth()
    assert truth_pyro.shape == (1, 1, mock_sim_data.time.shape[0])
