"""Verify the SensorSim experiment export contains nominal metadata."""

from pathlib import Path

import numpy as np
import pytest

import pyvale.sensorsim as sens
from pyvale.verif import pointsensscalar


@pytest.mark.parametrize(
    ("workers", "parallel_mode"),
    [
        (None, sens.EExpSimPara.ALL),
        (2, sens.EExpSimPara.ALL),
        (2, sens.EExpSimPara.SPLIT),
    ],
)
def test_experiment_archive_saves_nominal_positions_and_requested_truth(
    tmp_path: Path, workers: int | None, parallel_mode: sens.EExpSimPara
) -> None:
    simulation = pointsensscalar.simdata_2d_analytic()
    nominal_positions = np.array([[0.25, 0.25, 0.0], [0.5, 0.5, 0.0]])
    sensor_data = sens.SensorData(positions=nominal_positions)
    sensor = pointsensscalar.sens_array_noerrs(simulation, sensor_data, sens.EDim.TWOD)
    position_offset = np.array([[0.01, 0.0, 0.0], [0.01, 0.0, 0.0]])
    sensor.set_error_chain(
        [
            sens.ErrSysField(
                sensor.get_field(), sens.ErrFieldData(pos_offset_xyz=position_offset)
            )
        ]
    )
    simulator = sens.ExperimentSimulator(
        {"sim": simulation},
        {"temp": sensor},
        sens.ExpSimOpts(workers=workers, para=parallel_mode),
        sens.ExpSimSaveKeys(truth="truth", pert_sens_times=None),
    )
    exp_data = simulator.run_experiments(num_exp_per_sim=2)
    archive_path = tmp_path / "sensor_export.npz"
    sens.save_exp_sim_data(archive_path, exp_data)
    loaded = sens.load_exp_sim_data(archive_path)

    np.testing.assert_array_equal(
        loaded[("sim", "temp", "sens_pos")], nominal_positions
    )
    np.testing.assert_allclose(
        loaded[("sim", "temp", "pert_sens_pos")],
        np.broadcast_to(nominal_positions + position_offset, (2, 2, 3)),
    )
    truth = np.broadcast_to(sensor.get_truth(), loaded[("sim", "temp", "meas")].shape)
    np.testing.assert_array_equal(loaded[("sim", "temp", "truth")], truth)
