# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import pytest
import numpy as np
import pyvale.verif as verif
from pyvale.sensorsim.enums import EDim, ERayMode
from pyvale.sensorsim.fieldscalar import FieldScalar
from pyvale.sensorsim.fieldvector import FieldVector
from pyvale.sensorsim.sensorsray import SensorsRay
from pyvale.sensorsim.errorsysindep import ErrSysOffset
from pyvale.sensorsim.errorrand import ErrRandGen
from pyvale.sensorsim.generatorsrandom import GenNormal


def test_ray_distance_plane() -> None:
    sim_data, data_gen = verif.scalar_linear_2d()

    # Ray origin at (5.0, 3.75, 50.0), pointing towards -Z (0, 0, -1)
    # The mesh is at Z = 0.0. Exact distance = 50.0 mm
    origin = np.array([[5.0, 3.75, 50.0]])
    direction = np.array([[0.0, 0.0, -1.0]])

    ray_sens = SensorsRay(
        sim_data=sim_data,
        ray_origins=origin,
        ray_directions=direction,
        max_distance=200.0,
        sample_times=sim_data.time,
        mode=ERayMode.DISTANCE,
    )

    truth = ray_sens.get_truth()
    assert truth.shape == (1, 1, sim_data.time.shape[0])
    assert np.allclose(truth[0, 0, :], 50.0, atol=1e-4)


def test_ray_surface_field_sampling() -> None:
    sim_data, data_gen = verif.scalar_linear_2d()

    # Target point at (5.0, 3.75, 0.0)
    origin = np.array([[5.0, 3.75, 50.0]])
    direction = np.array([[0.0, 0.0, -1.0]])

    field = FieldScalar(sim_data, "temperature", EDim.TWOD)

    ray_sens = SensorsRay(
        sim_data=sim_data,
        ray_origins=origin,
        ray_directions=direction,
        field=field,
        max_distance=200.0,
        sample_times=sim_data.time,
        mode=ERayMode.SURFACE_FIELD,
    )

    truth = ray_sens.get_truth()
    # Sample directly with point sensor at hit point (5.0, 3.75, 0.0)
    pt_sample = field.sample_field(
        np.array([[5.0, 3.75, 0.0]]), times=sim_data.time
    )
    assert np.allclose(truth[0, :, :], pt_sample[0, :, :], atol=1e-4)


def test_ray_dynamic_deforming_mesh() -> None:
    sim_data, data_gen = verif.scalar_linear_2d()

    # Create dynamic displacement field u_z(t) = 10.0 * t
    n_pts = sim_data.coords.shape[0]
    n_times = sim_data.time.shape[0]
    sim_data.node_vars["disp_x"] = np.zeros((n_pts, n_times))
    sim_data.node_vars["disp_y"] = np.zeros((n_pts, n_times))
    # Disp z increases linearly with time
    disp_z = np.outer(np.ones(n_pts), 10.0 * sim_data.time)
    sim_data.node_vars["disp_z"] = disp_z

    disp_field = FieldVector(
        sim_data, ("disp_x", "disp_y", "disp_z"), EDim.TWOD
    )

    origin = np.array([[5.0, 3.75, 50.0]])
    direction = np.array([[0.0, 0.0, -1.0]])

    ray_sens = SensorsRay(
        sim_data=sim_data,
        ray_origins=origin,
        ray_directions=direction,
        disp_field=disp_field,
        max_distance=200.0,
        sample_times=sim_data.time,
        mode=ERayMode.DISTANCE,
    )

    truth = ray_sens.get_truth()
    # At t=0, distance = 50.0; at t=1.0, mesh is at z=10.0 -> distance = 40.0
    expected_dist = 50.0 - 10.0 * sim_data.time
    assert np.allclose(truth[0, 0, :], expected_dist, atol=1e-4)


def test_ray_errors_and_miss() -> None:
    sim_data, data_gen = verif.scalar_linear_2d()

    # Ray pointing away (+Z) -> miss -> max_distance
    origin = np.array([[5.0, 3.75, 50.0]])
    direction = np.array([[0.0, 0.0, 1.0]])

    ray_sens = SensorsRay(
        sim_data=sim_data,
        ray_origins=origin,
        ray_directions=direction,
        max_distance=150.0,
        sample_times=sim_data.time,
        mode=ERayMode.DISTANCE,
    )

    truth = ray_sens.get_truth()
    assert np.allclose(truth[0, 0, :], 150.0)

    # Attach error chain
    err_chain = (
        ErrSysOffset(offset=0.2),
        ErrRandGen(GenNormal(std=0.05, mean=0.0)),
    )
    ray_sens.set_error_chain(err_chain)
    meas = ray_sens.sim_measurements()
    err_tot = ray_sens.get_errors_total()
    assert err_tot is not None
    assert np.allclose(meas, truth + err_tot)
