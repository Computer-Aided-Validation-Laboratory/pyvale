#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================
import pytest
from typing import Callable, Dict, Any
import numpy as np

# Pyvale imports
import pyvale.sensorsim as sens
import pyvale.verif.pointsens as pointsens
import pyvale.verif.pointsensscalar as pointsensscalar
import pyvale.verif.pointsensvector as pointsensvector
import pyvale.verif.pointsenstensor as pointsenstensor

#-------------------------------------------------------------------------------
# TODO: Tests
# - Analytic test cases
# - Gold for multi-physics experiments
# - Logic for vector and tensor rotations
# - Area averaging for sensors on different faces in 2D

# VECTOR/TENSOR FIELDS:
# - rotation of area averaging

#-------------------------------------------------------------------------------
# Regression gold tests

@pytest.mark.parametrize(
    "get_sensors",
    [
        pointsensscalar.sens_arrays_2d_dict,
        pointsensscalar.sens_arrays_2d_analytic_dict,
        pointsensscalar.sens_arrays_2d_analytic_nomesh_dict,
        pointsensscalar.sens_arrays_3d_dict,
        pointsensscalar.sens_arrays_3d_nomesh_dict,
        pointsensvector.sens_arrays_2d_dict,
        pointsensvector.sens_arrays_3d_dict,
        pointsenstensor.sens_arrays_2d_dict,
        pointsenstensor.sens_arrays_3d_dict,
    ],
    ids=[
        "scalar_2d",
        "scalar_2d_analytic",
        "scalar_2d_analytic_nomesh",
        "scalar_3d",
        "scalar_3d_nomesh",
        "vector_2d",
        "vector_3d",
        "tensor_2d",
        "tensor_3d",
    ],
)
def test_gold_combined(get_sensors: Callable[[], Dict[str, Any]]) -> None:
    sensors = get_sensors()
    fails = pointsens.check_gold_measurements(sensors)
    assert not fails, "\n".join(fails)

#-------------------------------------------------------------------------------
# Check that 'get_measurements' does not resample probability distributions

def check_get_meas(sens_dict: dict[str,sens.SensorArrayPoint]) -> list[str]:
    fails = []
    for ss in sens_dict:
        calc_meas = sens_dict[ss].calc_measurements()
        get_meas = sens_dict[ss].get_measurements()

        if not np.allclose(calc_meas, get_meas):
            fails.append(f"Get does not equal calc for: {ss}")

    return fails

@pytest.mark.parametrize(
    "get_sensors",
    [
        pointsensscalar.sens_arrays_2d_dict,
        pointsensscalar.sens_arrays_2d_analytic_dict,
        pointsensscalar.sens_arrays_2d_analytic_nomesh_dict,
        pointsensscalar.sens_arrays_3d_dict,
        pointsensscalar.sens_arrays_3d_nomesh_dict,
        pointsensvector.sens_arrays_2d_dict,
        pointsensvector.sens_arrays_3d_dict,
        pointsenstensor.sens_arrays_2d_dict,
        pointsenstensor.sens_arrays_3d_dict,
    ],
    ids=[
        "scalar_2d",
        "scalar_2d_analytic",
        "scalar_2d_analytic_nomesh",
        "scalar_3d",
        "scalar_3d_nomesh",
        "vector_2d",
        "vector_3d",
        "tensor_2d",
        "tensor_3d",
    ],
)
def test_get_meas_combined(get_sensors: Callable[[], Dict[str, Any]]) -> None:
    sensors = get_sensors()
    fails = check_get_meas(sensors)
    assert not fails, "\n".join(fails)

#-------------------------------------------------------------------------------





