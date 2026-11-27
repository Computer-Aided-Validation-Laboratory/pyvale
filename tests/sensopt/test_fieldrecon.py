# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Analytic checks for reconstruction on scattered sensor sites."""

import numpy as np
import pytest

import pyvale.sensopt as opt
import pyvale.sensorsim as sens


def test_nearest_reconstruction_preserves_value_axes() -> None:
    sensors = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    values = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    targets = np.array([[1.5, 0.0, 0.0], [0.25, 0.0, 0.0]])
    predicted = opt.FieldReconNearest().reconstruct(
        sensor_positions=sensors, sensor_values=values, target_positions=targets
    )
    np.testing.assert_array_equal(predicted, values[[1, 0]])


@pytest.mark.parametrize("volume", [False, True])
def test_linear_reconstruction_matches_affine_field(*, volume: bool) -> None:
    sensors = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    inside = [0.25, 0.25, 0.5]
    outside = [[1.0, 1.0, 2.0], [0.25, 0.25, 1.25]]
    if volume:
        sensors = np.vstack((sensors, [0.0, 0.0, 1.0]))
        inside = [0.25, 0.25, 0.6]
    targets = np.array([inside, *outside])
    coefficients = np.array([[1.0, 2.0], [3.0, -1.0], [2.0, 1.0]])
    values = sensors @ coefficients + 5.0
    predicted = opt.FieldReconLinear(fill_value=-99.0).reconstruct(
        sensor_positions=sensors, sensor_values=values, target_positions=targets
    )
    np.testing.assert_allclose(predicted[0], targets[0] @ coefficients + 5.0)
    np.testing.assert_array_equal(predicted[1:], -99.0)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (opt.EErrMetric.RMSE, np.sqrt(5.0)),
        (opt.EErrMetric.MAE, 2.0),
        (opt.EErrMetric.MAX_ERR, 3.0),
    ],
)
def test_field_objective_scores_arbitrary_targets(
    metric: opt.EErrMetric, expected: float
) -> None:
    objective = opt.ObjFieldRecon(
        target_positions=np.array([[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]]),
        target_values={"nominal": np.array([[0.0], [1.0]])},
        sensor_keys=("strain",),
        component_index=0,
        metric=metric,
    )
    losses = objective.eval(
        sensor_data={
            "strain": sens.SensorData(
                positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            )
        },
        experiment_data={
            ("nominal", "strain", sens.EExpSimKeys.MEASUREMENT): np.array(
                [[[[1.0]], [[4.0]]]]
            )
        },
        scenario_keys=("nominal",),
    )
    np.testing.assert_allclose(losses, [[expected]])
