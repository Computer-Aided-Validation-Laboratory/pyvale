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
import pyvale.mooseherder as mh
import pyvale.verif.pointsens as pointsens
import pyvale.verif.pointsensscalar as pointsensscalar
import pyvale.verif.pointsensvector as pointsensvector
import pyvale.verif.pointsenstensor as pointsenstensor
import pyvale.verif.pointsensmech as pointsensmech
import pyvale.verif.analyticsimdatafactory as asd
import pyvale.verif.analyticsimdatagenerator as asg

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
    ],
    ids=[
        "scalar_2d",
        "scalar_2d_analytic",
        "scalar_2d_analytic_nomesh",
        "scalar_3d",
        "scalar_3d_nomesh",
    ],
)
def test_gold_sens_scalar(get_sensors: Callable[[], Dict[str, Any]]) -> None:
    sensors = get_sensors()
    fails = pointsens.check_gold_measurements(sensors)
    assert not fails, "\n".join(fails)

@pytest.mark.parametrize(
    "get_sensors",
    [
        pointsensvector.sens_arrays_2d_dict,
        pointsensvector.sens_arrays_2d_analytic_dict,
        pointsensvector.sens_arrays_2d_analytic_nomesh_dict,
        pointsensvector.sens_arrays_3d_dict,
        pointsensvector.sens_arrays_3d_nomesh_dict,

    ],
    ids=[
        "vector_2d",
        "vector_2d_analytic",
        "vector_2d_analytic_nomesh",
        "vector_3d",
        "vector_3d_nomesh",
    ],
)
def test_gold_sens_vector(get_sensors: Callable[[], Dict[str, Any]]) -> None:
    sensors = get_sensors()
    fails = pointsens.check_gold_measurements(sensors)
    assert not fails, "\n".join(fails)


@pytest.mark.parametrize(
    "get_sensors",
    [
        pointsenstensor.sens_arrays_2d_dict,
        pointsenstensor.sens_arrays_2d_analytic_dict,
        pointsenstensor.sens_arrays_2d_analytic_nomesh_dict,
        pointsenstensor.sens_arrays_3d_dict,
        pointsenstensor.sens_arrays_3d_nomesh_dict,

    ],
    ids=[
        "tensor_2d",
        "tensor_2d_analytic",
        "tensor_2d_analytic_nomesh",
        "tensor_3d",
        "tensor_3d_nomesh",
    ],
)
def test_gold_sens_tensor(get_sensors: Callable[[], Dict[str, Any]]) -> None:
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
    ],
    ids=[
        "scalar_2d",
        "scalar_2d_analytic",
        "scalar_2d_analytic_nomesh",
        "scalar_3d",
        "scalar_3d_nomesh",
    ],
)
def test_get_meas_scalar(get_sensors: Callable[[], Dict[str, Any]]) -> None:
    sensors = get_sensors()
    fails = check_get_meas(sensors)
    assert not fails, "\n".join(fails)

@pytest.mark.parametrize(
    "get_sensors",
    [
        pointsensvector.sens_arrays_2d_dict,
        pointsensvector.sens_arrays_2d_analytic_dict,
        pointsensvector.sens_arrays_2d_analytic_nomesh_dict,
        pointsensvector.sens_arrays_3d_dict,
        pointsensvector.sens_arrays_3d_nomesh_dict,
    ],
    ids=[
        "vector_2d",
        "vector_2d_analytic",
        "vector_2d_analytic_nomesh",
        "vector_3d",
        "vector_3d_nomesh",
    ],
)
def test_get_meas_vector(get_sensors: Callable[[], Dict[str, Any]]) -> None:
    sensors = get_sensors()
    fails = check_get_meas(sensors)
    assert not fails, "\n".join(fails)

@pytest.mark.parametrize(
    "get_sensors",
    [
        pointsenstensor.sens_arrays_2d_dict,
        pointsenstensor.sens_arrays_2d_analytic_dict,
        pointsenstensor.sens_arrays_2d_analytic_nomesh_dict,
        pointsenstensor.sens_arrays_3d_dict,
        pointsenstensor.sens_arrays_3d_nomesh_dict,
    ],
    ids=[
        "tensor_2d",
        "tensor_2d_analytic",
        "tensor_2d_analytic_nomesh",
        "tensor_3d",
        "tensor_3d_nomesh",
    ],
)
def test_get_meas_tensor(get_sensors: Callable[[], Dict[str, Any]]) -> None:
    sensors = get_sensors()
    fails = check_get_meas(sensors)
    assert not fails, "\n".join(fails)

#-------------------------------------------------------------------------------
# Analytic field comparison tests

def analytic_interp_scalar(sim_data: mh.SimData,
                           analytic_gen: asg.AnalyticSimDataGen) -> list[str]:

    sens_data_dict = pointsensscalar.sens_data_2d_dict(sim_data)

    fails = []
    for ss in sens_data_dict:
        sens_array = pointsensscalar.sens_array_noerrs(sim_data,
                                                       sens_data_dict[ss],
                                                       elem_dims=2)
        meas_sens = sens_array.get_measurements()

        sens_pos = sens_data_dict[ss].positions
        sens_times = sens_data_dict[ss].sample_times

        # dict[str,np.ndarray]
        meas_analytic = analytic_gen.evaluate_all_fields_truth(sens_pos,
                                                              sens_times)

        for mm in meas_analytic:
            comp_ind = sens_array.get_field().get_component_index(mm)
            if not np.allclose(meas_analytic[mm],meas_sens[:,comp_ind,:]):
                fails.append(f"SENSOR DATA: {ss}, FIELD: {mm}")

    if fails:
        fails.insert(0,"Analytic interp does not match for:")

    return fails


def test_analytic_interp_scalar_2d() -> None:
    (sim_data,analytic_gen) = asd.scalar_linear_2d()
    fails = analytic_interp_scalar(sim_data,analytic_gen)
    assert not fails, "\n".join(fails)

def test_analytic_interp_scalar_nomesh_2d() -> None:
    (sim_data,analytic_gen) = asd.scalar_linear_2d()
    sim_data.connect = None
    fails = analytic_interp_scalar(sim_data,analytic_gen)
    assert not fails, "\n".join(fails)


def analytic_interp_vector(sim_data: mh.SimData,
                           analytic_gen: asg.AnalyticSimDataGen) -> list[str]:

    sens_data_dict = pointsensmech.sens_data_2d_dict(sim_data)

    fails = []
    for ss in sens_data_dict:
        sens_array = pointsensvector.sens_array_2d_noerrs(sim_data,
                                                          sens_data_dict[ss])
        meas_sens = sens_array.get_measurements()

        sens_pos = sens_data_dict[ss].positions
        sens_times = sens_data_dict[ss].sample_times

        # dict[str,np.ndarray]
        meas_analytic = analytic_gen.evaluate_all_fields_truth(sens_pos,
                                                              sens_times)

        for mm in meas_analytic:
            comp_ind = sens_array.get_field().get_component_index(mm)
            if not np.allclose(meas_analytic[mm],meas_sens[:,comp_ind,:]):
                fails.append(f"SENSOR DATA: {ss}, FIELD: {mm}")

    if fails:
        fails.insert(0,"Analytic interp does not match.")

    return fails

def test_analytic_interp_vector_2d() -> None:
    (sim_data,analytic_gen) = asd.vector_linear_2d()
    fails = analytic_interp_vector(sim_data,analytic_gen)
    assert not fails, "\n".join(fails)

def test_analytic_interp_vector_nomesh_2d() -> None:
    (sim_data,analytic_gen) = asd.vector_linear_2d()
    sim_data.connect = None
    fails = analytic_interp_vector(sim_data,analytic_gen)
    assert not fails, "\n".join(fails)



def analytic_interp_tensor(sim_data: mh.SimData,
                           analytic_gen: asg.AnalyticSimDataGen) -> list[str]:

    sens_data_dict = pointsensmech.sens_data_2d_dict(sim_data)

    fails = []
    for ss in sens_data_dict:
        sens_array = pointsenstensor.sens_array_2d_noerrs(sim_data,
                                                          sens_data_dict[ss])
        meas_sens = sens_array.get_measurements()

        sens_pos = sens_data_dict[ss].positions
        sens_times = sens_data_dict[ss].sample_times

        # dict[str,np.ndarray]
        meas_analytic = analytic_gen.evaluate_all_fields_truth(sens_pos,
                                                              sens_times)

        for mm in meas_analytic:
            comp_ind = sens_array.get_field().get_component_index(mm)
            if not np.allclose(meas_analytic[mm],meas_sens[:,comp_ind,:]):
                fails.append(f"SENSDAT:{ss},FIELD:{mm},{meas_analytic[mm].shape=},{meas_sens[:,comp_ind,:].shape=}")

    if fails:
        fails.insert(0,"Analytic interp does not match.")

    return fails

def test_analytic_interp_tensor_2d() -> None:
    (sim_data,analytic_gen) = asd.tensor_linear_2d()
    fails = analytic_interp_tensor(sim_data,analytic_gen)
    assert not fails, "\n".join(fails)

def test_analytic_interp_tensor_nomesh_2d() -> None:
    (sim_data,analytic_gen) = asd.tensor_linear_2d()
    sim_data.connect = None
    fails = analytic_interp_tensor(sim_data,analytic_gen)
    assert not fails, "\n".join(fails)


