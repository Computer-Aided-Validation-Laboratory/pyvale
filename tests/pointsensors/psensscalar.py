#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================
from pathlib import Path
import numpy as np
import mooseherder as mh
import pyvale as pyv
import psensconst

def simdata_2d() -> mh.SimData:
    data_path = pyv.DataSet.thermal_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=None)
    return sim_data


def simdata_3d() -> mh.SimData:
    data_path = pyv.DataSet.thermal_3d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=None)
    return sim_data


def sens_pos_2d() -> dict[str,np.ndarray]:
    sim_dims = pyv.get_sim_dims(simdata_2d())
    sens_pos = {}

    x_lims = sim_dims["x"]
    y_lims = sim_dims["y"]
    z_lims = (0,0)

    n_sens = (4,1,1)
    sens_pos[""] = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    n_sens = (2,2,1)
    sens_pos_2 = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    return sens_pos


def sens_pos_3d() -> dict[str,np.ndarray]:
    sim_dims = pyv.get_sim_dims(simdata_3d())

    sens_pos = {}

    n_sens = (1,4,1)
    x_lims = (sim_dims["x"][1],sim_dims["x"][1])
    y_lims = sim_dims["y"]
    z_lims = sim_dims["z"]
    sens_pos["line-y-xy"] = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    n_sens = (1,4,1)
    x_lims = (9.4,9.4)
    y_lims = sim_dims["y"]
    z_lims = (sim_dims["z"][1],sim_dims["z"][1])
    sens_pos["line-y-yz"] = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    return sens_pos


def samp_times(sim_data: mh.SimData) -> dict[str, None | np.ndarray]:
    sim_dims = pyv.get_sim_dims(sim_data)
    sample_times = {}

    sample_times["sim"] = None
    sample_times["user"] = np.linspace(0.0,sim_dims["t"][1],50)

    return sample_times


# NOTE: this function generates all sensor data combinations
def sens_data_list(sim_data: mh.SimData,
                  sens_pos: list[np.ndarray]) -> list[pyv.SensorData]:
    sample_times = samp_times(sim_data)

    sens_data = []
    tags = []
    for pp in sens_pos:

        for tt in sample_times:

            sens_data.append(pyv.SensorData(
                positions=sens_pos[pp],
                sample_times=sample_times[tt],
            ))

            tags.append({"pos": pp,
                         "time":tt})

    return sens_data


def sens_data_list_2d() -> list[pyv.SensorData]:
    return sens_data_list(simdata_2d(),sens_pos_2d())


def sens_data_list_3d() -> list[pyv.SensorData]:
    return sens_data_list(simdata_3d(),sens_pos_3d())


def sens_data_2d(case_ind: int = 0) -> pyv.SensorData:
    return sens_data_list_2d()[case_ind]


def sens_data_3d(case_ind: int = 0) -> pyv.SensorData:
    return sens_data_list_3d()[case_ind]


def sens_2d(sens_data: pyv.SensorData) -> pyv.SensorArrayPoint:
    sim_data = simdata_2d()
    descriptor = pyv.SensorDescriptorFactory.temperature_descriptor()
    field = pyv.FieldScalar(sim_data,
                            field_key="temperature",
                            elem_dims=2)
    return pyv.SensorArrayPoint(sens_data,
                                field,
                                descriptor)

def sens_3d(sens_data: pyv.SensorData) -> pyv.SensorArrayPoint:
    sim_data = simdata_3d()
    descriptor = pyv.SensorDescriptorFactory.temperature_descriptor()
    field = pyv.FieldScalar(sim_data,
                            field_key="temperature",
                            elem_dims=3)
    return pyv.SensorArrayPoint(sens_data,
                                field,
                                descriptor)

# TODO: need to have different cases of error integrators
def err_chains() -> dict[str,list[pyv.IErrCalculator]]:
    err_cases = {}

    # CASE 1: basic statistical errors
    chain_basic = []
    chain_basic.append(pyv.ErrSysOffset(offset=-1.0))
    chain_basic.append(pyv.ErrSysUnif(low=-1.0,
                                      high=1.0,
                                      seed=psensconst.GOLD_SEED))
    chain_basic.append(pyv.ErrSysUnifPercent(low_percent=-1.0,
                                             high_percent=1.0,
                                             seed=psensconst.GOLD_SEED))
    chain_basic.append(pyv.ErrRandNorm(std=1.0,
                                       seed=psensconst.GOLD_SEED))
    chain_basic.append(pyv.ErrRandNormPercent(std_percent=1.0,
                                              seed=psensconst.GOLD_SEED))
    err_cases["basic"] = chain_basic

    # CASE 2: as above but using the generator interface
    chain_gen = []
    chain_basic.append(pyv.ErrSysOffset(offset=-1.0))
    chain_gen.append(pyv.ErrSysGen(
        pyv.GenUniform(low=-1.0,high=1.0,seed=psensconst.GOLD_SEED)))
    chain_gen.append(pyv.ErrSysGenPercent(
        pyv.GenUniform(low=-1.0,high=1.0,seed=psensconst.GOLD_SEED)))
    chain_gen.append(pyv.ErrRandGen(
        pyv.GenUniform(std=1.0,seed=psensconst.GOLD_SEED)))
    chain_gen.append(pyv.ErrRandGenPercent(
        pyv.GenUniform(std=1.0,seed=psensconst.GOLD_SEED)))

    err_cases["basic-gen"] = chain_gen

    # CASE 3: dependent errors


    # CASE 4: field errors

    # CASE 5: all error types stacked


def gen_gold_2d(save_path: Path) -> None:
    sens_data = sens_data_list_2d()
