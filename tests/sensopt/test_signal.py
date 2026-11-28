# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Check ordered and pooled signal criteria."""

import numpy as np

import pyvale.sensoropt as opt
import pyvale.sensorsim as sens


def test_signal_reduction_order_changes_the_score() -> None:
    history = np.array([[[[10.0, 1.0]], [[1.0, 10.0]]]])
    experiment_data = {("nominal", "strain", sens.EExpSimKeys.TRUTH): history}
    component = opt.SignalReduction(
        axis=opt.ESignalAxis.COMPONENT, method=opt.ESignalReduce.L2
    )
    position = opt.SignalReduction(
        axis=opt.ESignalAxis.POSITION, method=opt.ESignalReduce.MAX
    )
    time = opt.SignalReduction(axis=opt.ESignalAxis.TIME, method=opt.ESignalReduce.MIN)
    for steps, expected in (
        ((component, position, time), -10.0),
        ((component, time, position), -1.0),
    ):
        objective = opt.ObjSignal(sensor_keys=("strain",), reductions=steps)
        np.testing.assert_array_equal(
            objective.eval(
                sensor_data={},
                experiment_data=experiment_data,
                scenario_keys=("nominal",),
            ),
            [[expected]],
        )


def test_signal_presets_pool_arrays_by_reading_count() -> None:
    experiment_data = {
        ("nominal", "first", sens.EExpSimKeys.TRUTH): np.array([[[[-5.0]]]]),
        ("nominal", "second", sens.EExpSimKeys.TRUTH): np.array([[[[1.0, 2.0, 3.0]]]]),
    }
    keys = ("first", "second")
    for objective, expected in (
        (opt.ObjSignalRMS(sensor_keys=keys), -np.sqrt(39 / 4)),
        (opt.ObjSignalPeakAbs(sensor_keys=keys), -5.0),
    ):
        np.testing.assert_allclose(
            objective.eval(
                sensor_data={},
                experiment_data=experiment_data,
                scenario_keys=("nominal",),
            ),
            [[expected]],
        )
