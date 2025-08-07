#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================
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
applied to mechanical fields (displacement/strain) for testing vector and tensor
field point sensors.
"""

def simdata_mech_2d() -> mh.SimData:
    data_path = dataset.mechanical_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    sim_data = sens.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=("disp_x","disp_y"))
    return sim_data


def simdata_mech_3d() -> mh.SimData:
    data_path = dataset.element_case_output_path(dataset.EElemTest.HEX20)
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    field_comps = ("disp_x","disp_y","disp_z")
    sim_data = sens.scale_length_units(scale=1000.0,
                                        sim_data=sim_data,
                                        disp_comps=field_comps)
    return sim_data

def sens_pos_2d() -> dict[str,np.ndarray]:
    sim_dims = sens.SimTools.get_sim_dims(simdata_mech_2d())
    sens_pos = {}

    x_lims = sim_dims["x"]
    y_lims = sim_dims["y"]
    z_lims = (0,0)

    n_sens = (1,4,1)
    sens_pos["line-4"] = sens.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    n_sens = (2,3,1)
    sens_pos["grid-23"] = sens.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    return sens_pos


def sens_pos_3d() -> dict[str,np.ndarray]:
    sens_pos = {}
    sens_pos["cent-cube"] = np.array(((5.0,0.0,5.0),    # xz
                                      (5.0,10.0,5.0),   # xz
                                      (5.0,5.0,0.0),    # xy
                                      (5.0,5.0,10.0),   # xy
                                      (0.0,5.0,5.0),    # yz
                                      (10.0,5.0,5.0),)) # yz
    return sens_pos

def sens_pos_2d_lock(sens_pos: np.ndarray) -> dict[str,np.ndarray]:
    pos_lock = {}
    (xx,yy,zz) = (0,1,2)

    lock = np.full_like(sens_pos,False,dtype=bool)
    lock[:,zz] = True # lock z
    pos_lock["line-4"] = None

    lock = np.full_like(sens_pos,False,dtype=bool)
    lock[:,zz] = True # lock z
    pos_lock["grid-23"] = None

    return pos_lock


def sens_pos_3d_lock(sens_pos: np.ndarray) -> dict[str,np.ndarray]:
    pos_lock = {}
    (xx,yy,zz) = (0,1,2)

    lock = np.full_like(sens_pos,False,dtype=bool)
    lock[0,yy] = True
    lock[1,yy] = True
    lock[2,zz] = True
    lock[3,zz] = True
    lock[4,xx] = True
    lock[5,xx] = True
    pos_lock["cent-cube"] = lock

    return pos_lock


def sens_data_2d_dict() -> dict[str,sens.SensorData]:
    return psens.sens_data_dict(simdata_mech_2d(),sens_pos_2d())


def sens_data_3d_dict() -> dict[str,sens.SensorData]:
    return psens.sens_data_dict(simdata_mech_3d(),sens_pos_3d())


def err_chain_field(field: sens.IField,
                    sens_pos: np.ndarray,
                    samp_times: np.ndarray | None,
                    pos_lock: np.ndarray | None,
                    ) -> list[sens.IErrCalculator]:

    if samp_times is None:
        samp_times = field.get_time_steps()

    pos_offset_xyz = np.array((1.0,1.0,-1.0),dtype=np.float64)
    pos_offset_xyz = np.tile(pos_offset_xyz,(sens_pos.shape[0],1))

    time_offset = np.full((samp_times.shape[0],),0.1)

    pos_rand = sens.GenNormal(std=1.0,
                             mean=0.0,
                             seed=psensconst.GOLD_SEED) # units = mm
    time_rand = sens.GenNormal(std=0.1,
                              mean=0.0,
                              seed=psensconst.GOLD_SEED) # units = s
    ang_rand = sens.GenUniform(low=-1.0,
                              high=1.0,
                              seed=psensconst.GOLD_SEED)

    field_err_data = sens.ErrFieldData(
        pos_offset_xyz=pos_offset_xyz,
        time_offset=time_offset,
        pos_rand_xyz=(pos_rand,pos_rand,pos_rand),
        ang_rand_zyx=(ang_rand,ang_rand,ang_rand),
        time_rand=time_rand,
        pos_lock_xyz=pos_lock,
    )

    err_chain = []
    err_chain.append(sens.ErrSysField(field,
                                     field_err_data))
    return err_chain


def err_chain_field_dep(field: sens.IField,
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

    angle_offset = np.ones_like(sens_pos)
    angle_error_data = sens.ErrFieldData(ang_offset_zyx=angle_offset)

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
    err_chain.append(sens.ErrSysField(field,
                                    angle_error_data,
                                    sens.EErrDep.DEPENDENT))
    err_chain.append(sens.ErrSysField(field,
                                    angle_error_data,
                                    sens.EErrDep.DEPENDENT))
    return err_chain


def err_chain_2d_dict(field: sens.IField,
                      sens_pos: np.ndarray,
                      samp_times: np.ndarray | None,
                      pos_lock: np.ndarray | None
                      ) -> dict[str,list[sens.IErrCalculator]]:
    err_cases = {}
    err_cases["none"] = None
    err_cases["basic"] = psens.err_chain_basic()
    err_cases["basic-gen"] = psens.err_chain_gen()
    err_cases["field"] = err_chain_field(field,sens_pos,samp_times,pos_lock)
    err_cases["field-dep"] = err_chain_field_dep(field,sens_pos,samp_times,pos_lock)

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
    err_cases["field"] = err_chain_field(field,sens_pos,samp_times,pos_lock)
    err_cases["field-dep"] = err_chain_field_dep(field,sens_pos,samp_times,pos_lock)

    # This has to be last so when we chain all errors together the saturation
    # error is the last thing that happens
    err_cases["basic-dep"] = psens.err_chain_dep()

    err_cases["all"] = psens.err_chain_all(err_cases)

    return err_cases