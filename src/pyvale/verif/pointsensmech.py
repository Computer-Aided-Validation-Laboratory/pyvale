#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================
import numpy as np
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.verif.pointsens as pointsens
import pyvale.verif.pointsensconst as pointsensconst
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
    sim_data = mh.ExodusLoader(data_path).load_all_sim_data()
    sim_data = sens.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_keys=("disp_x","disp_y"))
    return sim_data

def simdata_mesh_2d_nomesh() -> mh.SimData:
    sim_data = simdata_mech_2d()
    sim_data.connect = None
    return sim_data

def simdata_mech_3d() -> mh.SimData:
    data_path = dataset.element_case_output_path(dataset.EElemTest.HEX20)
    sim_data = mh.ExodusLoader(data_path).load_all_sim_data()
    field_comps = ("disp_x","disp_y","disp_z")
    sim_data = sens.scale_length_units(scale=1000.0,
                                        sim_data=sim_data,
                                        disp_keys=field_comps)
    return sim_data

def simdata_mech_3d_nomesh() -> mh.SimData:
    sim_data = simdata_mech_3d()
    sim_data.connect = None
    return sim_data


def sens_pos_2d(sim_data: mh.SimData) -> dict[str,np.ndarray]:
    sim_dims = sens.simtools.get_sim_dims(sim_data)
    sens_pos = {}

    x_lims = sim_dims["x"]
    y_lims = sim_dims["y"]
    z_lims = (0,0)

    n_sens = (1,4,1)
    sens_pos["line-4"] = sens.gen_pos_grid_inside(n_sens,x_lims,y_lims,z_lims)

    n_sens = (2,3,1)
    sens_pos["grid-23"] = sens.gen_pos_grid_inside(n_sens,x_lims,y_lims,z_lims)

    return sens_pos


def sens_pos_3d(sim_data: mh.SimData) -> dict[str,np.ndarray]:
    sim_dims = sens.simtools.get_sim_dims(sim_data)
    (x_min,x_max) = sim_dims["x"]
    (y_min,y_max) = sim_dims["y"]
    (z_min,z_max) = sim_dims["z"]
    x_len = x_max-x_min
    y_len = y_max-y_min
    z_len = z_max-z_min

    sens_pos = {}


    sens_pos["cent-cube"] = np.array((
        (x_min+x_len/2, y_min,          z_min+z_len/2), # xz
        (x_min+x_len/2, y_min+y_len,    z_min+z_len/2), # xz
        (x_min+x_len/2, y_min+y_len/2,  z_min),         # xy
        (x_min+x_len/2, y_min+y_len/2,  z_min+z_len),   # xy
        (x_min,         y_min+y_len/2,  z_min+z_len/2), # yz
        (x_min+x_len,   y_min+y_len/2,  z_min+z_len/2), # yz
    ))

    # check = np.array(((5.0,0.0,5.0),    # xz
    #                 (5.0,10.0,5.0),   # xz
    #                 (5.0,5.0,0.0),    # xy
    #                 (5.0,5.0,10.0),   # xy
    #                 (0.0,5.0,5.0),    # yz
    #                 (10.0,5.0,5.0),)) # yz

    # assert np.allclose(check,sens_pos["cent-cube"]), "Cube coords wrong in mech.sens_pos_3d"

    return sens_pos


def sens_pos_2d_lock(sens_pos: np.ndarray) -> dict[str,np.ndarray]:
    pos_lock = {}
    (xx,yy,zz) = (0,1,2)

    lock = np.full_like(sens_pos,False,dtype=bool)
    lock[:,zz] = True # lock z
    pos_lock["line-4"] = lock

    lock = np.full_like(sens_pos,False,dtype=bool)
    lock[:,zz] = True # lock z
    pos_lock["grid-23"] = lock

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


def sens_data_2d_dict(sim_data: mh.SimData) -> dict[str,sens.SensorData]:
    return pointsens.sens_data_dict(sim_data,sens_pos_2d(sim_data))


def sens_data_3d_dict(sim_data: mh.SimData) -> dict[str,sens.SensorData]:
    return pointsens.sens_data_dict(sim_data,sens_pos_3d(sim_data))


def err_chain_field(field: sens.IField,
                    sens_pos: np.ndarray,
                    samp_times: np.ndarray | None,
                    pos_lock: np.ndarray | None,
                    ) -> list[sens.IErrSimulator]:

    if samp_times is None:
        samp_times = field.get_time_steps()

    pos_offset_xyz = np.array((0.5,0.5,-0.5),dtype=np.float64)
    pos_offset_xyz = np.tile(pos_offset_xyz,(sens_pos.shape[0],1))

    time_offset = np.full((samp_times.shape[0],),0.1)

    pos_rand = sens.GenNormal(std=0.1,
                             mean=0.0,
                             seed=pointsensconst.GOLD_SEED) # units = mm
    time_rand = sens.GenNormal(std=0.1,
                              mean=0.0,
                              seed=pointsensconst.GOLD_SEED) # units = s
    ang_rand = sens.GenUniform(low=-1.0,
                              high=1.0,
                              seed=pointsensconst.GOLD_SEED)

    field_err_data = sens.ErrFieldData(
        pos_offset_xyz=pos_offset_xyz,
        time_offset=time_offset,
        pos_rand_xyz=(pos_rand,pos_rand,pos_rand),
        ang_rand_zyx=(ang_rand,ang_rand,ang_rand),
        time_rand=time_rand,
        pos_lock_xyz=pos_lock,
    )

    err_chain = [sens.ErrSysField(field,field_err_data),]
    return err_chain


def err_chain_field_dep(field: sens.IField,
                        sens_pos: np.ndarray,
                        samp_times: np.ndarray | None,
                        pos_lock: np.ndarray | None,
                        ) -> list[sens.IErrSimulator]:

    if samp_times is None:
        samp_times = field.get_time_steps()

    time_offset = 0.1*np.ones_like(samp_times)
    time_error_data = sens.ErrFieldData(time_offset=time_offset)

    pos_offset = -0.2*np.ones_like(sens_pos)
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
                      ) -> dict[str,list[sens.IErrSimulator]]:
    err_cases = {}
    err_cases["none"] = None
    err_cases["basic"] = pointsens.err_chain_basic()
    err_cases["basic-gen"] = pointsens.err_chain_gen()
    err_cases["field"] = err_chain_field(field,
                                         sens_pos,
                                         samp_times,
                                         pos_lock)
    err_cases["field-dep"] = err_chain_field_dep(field,
                                                 sens_pos,
                                                 samp_times,
                                                 pos_lock)

    # This has to be last so when we chain all errors together the saturation
    # error is the last thing that happens
    err_cases["basic-dep"] = pointsens.err_chain_dep()

    err_cases["all"] = pointsens.err_chain_all(err_cases)

    return err_cases


def err_chain_3d_dict(field: sens.IField,
                      sens_pos: np.ndarray,
                      samp_times: np.ndarray | None,
                      pos_lock: np.ndarray | None
                      ) -> dict[str,list[sens.IErrSimulator]]:
    err_cases = {}
    err_cases["none"] = None
    err_cases["basic"] = pointsens.err_chain_basic()
    err_cases["basic-gen"] = pointsens.err_chain_gen()
    err_cases["field"] = err_chain_field(field,
                                         sens_pos,
                                         samp_times,
                                         pos_lock)
    err_cases["field-dep"] = err_chain_field_dep(field,
                                                 sens_pos,
                                                 samp_times,
                                                 pos_lock)

    # This has to be last so when we chain all errors together the saturation
    # error is the last thing that happens
    err_cases["basic-dep"] = pointsens.err_chain_dep()

    err_cases["all"] = pointsens.err_chain_all(err_cases)

    return err_cases
