#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================
import copy
import numpy as np

import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.verif.psens as psens
import pyvale.verif.psensconst as psensconst
import pyvale.dataset as dataset

"""
DEVELOPER VERIFICATION MODULE
--------------------------------------------------------------------------------
This module contains developer utility functions used for verification testing
of the point sensor simulation toolbox in pyvale.

Specifically, this module contains functions used for testing point sensors
applied to scalar fields.
"""

# TODO: fix position locking for 3D field errors

def simdata_2d() -> mh.SimData:
    data_path = dataset.thermal_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    sim_data = sens.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=None)
    return sim_data


def simdata_3d() -> mh.SimData:
    data_path = dataset.thermal_3d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    sim_data = sens.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=None)
    return sim_data


def sens_pos_2d() -> dict[str,np.ndarray]:
    sim_dims = sens.SimTools.get_sim_dims(simdata_2d())
    sens_pos = {}

    x_lims = sim_dims["x"]
    y_lims = sim_dims["y"]
    z_lims = (0,0)

    n_sens = (4,1,1)
    sens_pos["line-4"] = sens.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    n_sens = (2,2,1)
    sens_pos["grid-22"] = sens.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    return sens_pos


def sens_pos_3d() -> dict[str,np.ndarray]:
    sim_dims = sens.SimTools.get_sim_dims(simdata_3d())

    sens_pos = {}

    n_sens = (1,4,1)
    x_lims = (sim_dims["x"][1],sim_dims["x"][1])
    y_lims = sim_dims["y"]
    z_lims = sim_dims["z"]
    sens_pos["line-y-xy"] = sens.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    n_sens = (1,4,1)
    x_lims = (9.4,9.4)
    y_lims = sim_dims["y"]
    z_lims = (sim_dims["z"][1],sim_dims["z"][1])
    sens_pos["line-y-yz"] = sens.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    return sens_pos


def sens_pos_2d_lock(sens_pos: np.ndarray) -> dict[str,np.ndarray]:
    pos_lock = {}

    lock = np.full_like(sens_pos,False,dtype=bool)
    lock[:,2] = True # lock z
    pos_lock["line-4"] = None

    lock = np.full_like(sens_pos,False,dtype=bool)
    lock[:,2] = True # lock z
    pos_lock["grid-22"] = None

    return pos_lock


def sens_pos_3d_lock(sens_pos: np.ndarray) -> dict[str,np.ndarray]:
    pos_lock = {}

    lock = np.full_like(sens_pos,False,dtype=bool)
    lock[:,2] = True # lock z
    pos_lock["line-y-xy"] = lock

    lock = np.full_like(sens_pos,False,dtype=bool)
    lock[:,0] = True # lock x
    pos_lock["line-y-yz"] = None

    return pos_lock


def sens_data_2d_dict() -> dict[str,sens.SensorData]:
    return psens.sens_data_dict(simdata_2d(),sens_pos_2d())


def sens_data_3d_dict() -> dict[str,sens.SensorData]:
    return psens.sens_data_dict(simdata_3d(),sens_pos_3d())


def err_chain_sfield(field: sens.IField,
                    sens_pos: np.ndarray,
                    samp_times: np.ndarray | None,
                    pos_lock: np.ndarray | None,
                    ) -> list[sens.IErrCalculator]:

    if samp_times is None:
        samp_times = field.get_time_steps()

    pos_offset_xyz = np.array((1.0,1.0,1.0),dtype=np.float64)
    pos_offset_xyz = np.tile(pos_offset_xyz,(sens_pos.shape[0],1))

    time_offset = np.full((samp_times.shape[0],),0.1)

    pos_rand = sens.GenNormal(std=1.0,
                             mean=0.0,
                             seed=psensconst.GOLD_SEED) # units = mm
    time_rand = sens.GenNormal(std=0.1,
                              mean=0.0,
                              seed=psensconst.GOLD_SEED) # units = s

    field_err_data = sens.ErrFieldData(
        pos_offset_xyz=pos_offset_xyz,
        time_offset=time_offset,
        pos_rand_xyz=(pos_rand,pos_rand,pos_rand),
        time_rand=time_rand,
        pos_lock_xyz=pos_lock,
    )

    err_chain = []
    err_chain.append(sens.ErrSysField(field,
                                     field_err_data))
    return err_chain


def err_chain_sfield_dep(field: sens.IField,
                        sens_pos: np.ndarray,
                        samp_times: np.ndarray | None,
                        pos_lock: np.ndarray | None,
                        ) -> list[sens.IErrCalculator]:

    if samp_times is None:
        samp_times = field.get_time_steps()

    time_offset = 2.0*np.ones_like(samp_times)
    time_error_data = sens.ErrFieldData(time_offset=time_offset)

    pos_offset = -1.0*np.ones_like(sens_pos)
    pos_error_data = sens.ErrFieldData(pos_offset_xyz=pos_offset,
                                      pos_lock_xyz=pos_lock)

    err_chain = []
    err_chain.append(sens.ErrSysField(field,
                                    time_error_data,
                                    sens.EErrDep.DEPENDENT))
    err_chain.append(sens.ErrSysField(field,
                                    time_error_data,
                                    sens.EErrDep.DEPENDENT))

    err_chain.append(sens.ErrSysField(field,
                                    pos_error_data,
                                    sens.EErrDep.DEPENDENT))
    err_chain.append(sens.ErrSysField(field,
                                    pos_error_data,
                                    sens.EErrDep.DEPENDENT))
    return err_chain


def calib_assumed(signal: np.ndarray) -> np.ndarray:
    return 24.3*signal + 0.616

def calib_truth(signal: np.ndarray) -> np.ndarray:
    return -0.01897 + 25.41881*signal - 0.42456*signal**2 + 0.04365*signal**3

def err_chain_calib() -> list[sens.IErrCalculator]:
    signal_calib_range = np.array((0.0,6.0),dtype=np.float64)
    cal_err = sens.ErrSysCalibration(calib_assumed,
                                    calib_truth,
                                    signal_calib_range,
                                    n_cal_divs=10000)
    return [cal_err,]


def err_chain_2d_dict(field: sens.IField,
                      sens_pos: np.ndarray,
                      samp_times: np.ndarray | None,
                      pos_lock: np.ndarray | None
                      ) -> dict[str,list[sens.IErrCalculator]]:
    err_cases = {}
    err_cases["none"] = None
    err_cases["basic"] = psens.err_chain_basic()
    err_cases["basic-gen"] = psens.err_chain_gen()
    err_cases["field"] = err_chain_sfield(field,sens_pos,samp_times,pos_lock)
    err_cases["field-dep"] = err_chain_sfield_dep(field,sens_pos,samp_times,pos_lock)
    err_cases["calib"] = err_chain_calib()

     # This has to be last so when we chain all errors together the saturation
    # error is the last thing that happens
    err_cases["basic-dep"] = psens.err_chain_dep()

    err_cases["all"] = psens.err_chain_all(err_cases)

    return err_cases


def err_chain_3d_dict(field: sens.IField,
                      sens_pos: np.ndarray,
                      samp_times: np.ndarray | None,
                      pos_lock: np.ndarray | None
                      ) -> dict[str,list[sens.IErrCalculator]]:
    err_cases = {}
    err_cases["none"] = None
    err_cases["basic"] = psens.err_chain_basic()
    err_cases["basic-gen"] = psens.err_chain_gen()
    err_cases["field"] = err_chain_sfield(field,sens_pos,samp_times,pos_lock)
    err_cases["field-dep"] = err_chain_sfield_dep(field,sens_pos,samp_times,pos_lock)
    err_cases["calib"] = err_chain_calib()

    # This has to be last so when we chain all errors together the saturation
    # error is the last thing that happens
    err_cases["basic-dep"] = psens.err_chain_dep()

    err_cases["all"] = psens.err_chain_all(err_cases)

    return err_cases


def sens_noerrs(sim_data: mh.SimData,
                sens_data: sens.SensorData,
                elem_dims: int) -> sens.SensorArrayPoint:
    descriptor = sens.SensorDescriptorFactory.temperature_descriptor()
    field = sens.FieldScalar(sim_data,
                            field_key="temperature",
                            elem_dims=elem_dims)
    sens_array =  sens.SensorArrayPoint(sens_data,
                                       field,
                                       descriptor)
    return sens_array


def gen_sens_dict_2d(sim_data: mh.SimData,
                    sens_data_dict: dict[str, sens.SensorData],
                    tag: str
                    ) -> dict[str, sens.SensorArrayPoint]:

    sens_dict = {}
    for ss in sens_data_dict:
        sens_array = sens_noerrs(sim_data,
                                 sens_data_dict[ss],
                                 elem_dims=2)

        pos_lock = sens_pos_2d_lock(sens_data_dict[ss].positions)
        for kk in pos_lock:
            if kk in ss:
                pos_lock_key = kk
                break

        err_chain_dict = err_chain_2d_dict(sens_array.get_field(),
                                           sens_data_dict[ss].positions,
                                           sens_data_dict[ss].sample_times,
                                           pos_lock[pos_lock_key])

        for ee in err_chain_dict:
            key = f"{tag}_{ss}_err-{ee}"
            sens_dict[key] = copy.deepcopy(sens_array)

            if err_chain_dict[ee] is not None:
                err_int_opts = sens.ErrIntOpts()
                err_int = sens.ErrIntegrator(err_chain_dict[ee],
                                            sens_data_dict[ss],
                                            sens_dict[key].get_measurement_shape(),
                                            err_int_opts=err_int_opts)
                sens_dict[key].set_error_integrator(err_int)

    return sens_dict

def gen_sens_dict_3d(sim_data: mh.SimData,
                    sens_data_dict: dict[str,sens.SensorData],
                    tag: str,
                    ) -> dict[str,sens.SensorArrayPoint]:

    sens_dict = {}
    for ss in sens_data_dict:
        sens_array = sens_noerrs(sim_data,
                                 sens_data_dict[ss],
                                 elem_dims=3)

        pos_lock = sens_pos_3d_lock(sens_data_dict[ss].positions)
        for kk in pos_lock:
            if kk in ss:
                pos_lock_key = kk
                break

        err_chain_dict = err_chain_3d_dict(sens_array.get_field(),
                                           sens_data_dict[ss].positions,
                                           sens_data_dict[ss].sample_times,
                                           pos_lock[pos_lock_key])

        for ee in err_chain_dict:
            key = f"{tag}_{ss}_err-{ee}"
            sens_dict[key] = copy.deepcopy(sens_array)

            if err_chain_dict[ee] is not None:
                err_int_opts = sens.ErrIntOpts()
                err_int = sens.ErrIntegrator(err_chain_dict[ee],
                                            sens_data_dict[ss],
                                            sens_dict[key].get_measurement_shape(),
                                            err_int_opts=err_int_opts)
                sens_dict[key].set_error_integrator(err_int)

    return sens_dict



def sens_2d_dict() -> dict[str,sens.SensorArrayPoint]:
    sens_data_dict = sens_data_2d_dict()
    sim_data = simdata_2d()
    tag = "scal2d"
    return gen_sens_dict_2d(sim_data,sens_data_dict,tag)

def sens_3d_dict() -> dict[str,sens.SensorArrayPoint]:
    sens_data_dict = sens_data_3d_dict()
    sim_data = simdata_3d()
    tag = "scal3d"
    return gen_sens_dict_3d(sim_data,sens_data_dict,tag)





