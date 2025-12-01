#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================
import copy
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.verif.pointsensmech as pointsensmech
import pyvale.verif.analyticsimdatafactory as asd


"""
DEVELOPER VERIFICATION MODULE
--------------------------------------------------------------------------------
This module contains developer utility functions used for verification testing
of the point sensor simulation toolbox in pyvale.

Specifically, this module contains functions used for testing point sensors
applied to vector fields.
"""

# TODO
# - Calibration errors for vector fields

def simdata_vec_2d_analytic() -> mh.SimData:
    (sim_data,_) = asd.vector_linear_2d()
    return sim_data

def simdata_vec_2d_analytic_nomesh() -> mh.SimData:
    sim_data = simdata_vec_2d_analytic()
    sim_data.connect = None
    return sim_data


def sens_array_2d_noerrs(sim_data: mh.SimData,
                   sens_data: sens.SensorData,
                   spatial_dims: sens.EDim = sens.EDim.TWOD,
                   ) -> sens.SensorsPoint:
    descriptor = sens.DescriptorFactory.displacement()
    field = sens.FieldVector(sim_data,
                            comp_keys=("disp_x","disp_y"),
                            spatial_dims=spatial_dims)
    sens_array = sens.SensorsPoint(sens_data,
                                      field,
                                      descriptor)
    return sens_array


def sens_array_3d_noerrs(sim_data: mh.SimData,
                   sens_data: sens.SensorData,
                   spatial_dims: sens.EDim = sens.EDim.THREED
                   ) -> sens.SensorsPoint:
    descriptor = sens.DescriptorFactory.displacement()
    field = sens.FieldVector(sim_data,
                            comp_keys=("disp_x","disp_y","disp_z"),
                            spatial_dims=spatial_dims)
    sens_array =  sens.SensorsPoint(sens_data,
                                       field,
                                       descriptor)
    return sens_array

#-------------------------------------------------------------------------------
def gen_sens_arrays_2d_dict(sim_data: mh.SimData,
                            sens_data_dict: dict[str,sens.SensorData],
                            sens_tag: str,
                            ) -> dict[str,sens.SensorsPoint]:
    sens_dict = {}
    for ss in sens_data_dict:
        sens_array = sens_array_2d_noerrs(sim_data,sens_data_dict[ss])

        pos_lock = pointsensmech.sens_pos_2d_lock(sens_data_dict[ss].positions)
        for kk in pos_lock:
            if kk in ss:
                pos_lock_key = kk
                break

        err_chain_dict = pointsensmech.err_chain_2d_dict(sens_array.get_field(),
                                           sens_data_dict[ss].positions,
                                           sens_data_dict[ss].sample_times,
                                           pos_lock[pos_lock_key])

        for ee in err_chain_dict:
            tag = f"{sens_tag}_{ss}_err-{ee}"
            sens_dict[tag] = copy.deepcopy(sens_array)

            if err_chain_dict[ee] is not None:
                err_int_opts = sens.ErrIntOpts()
                sens_dict[tag].set_error_chain(err_chain_dict[ee],
                                               err_int_opts)

    return sens_dict


def gen_sens_arrays_3d_dict(sim_data: mh.SimData,
                            sens_data_dict: dict[str,sens.SensorData],
                            sens_tag: str,
                            ) -> dict[str,sens.SensorsPoint]:
    sens_dict = {}
    for ss in sens_data_dict:
        sens_array = sens_array_3d_noerrs(sim_data,sens_data_dict[ss])

        pos_lock = pointsensmech.sens_pos_3d_lock(sens_data_dict[ss].positions)
        for kk in pos_lock:
            if kk in ss:
                pos_lock_key = kk
                break

        err_chain_dict = pointsensmech.err_chain_3d_dict(sens_array.get_field(),
                                           sens_data_dict[ss].positions,
                                           sens_data_dict[ss].sample_times,
                                           pos_lock=pos_lock[pos_lock_key])

        for ee in err_chain_dict:
            tag = f"{sens_tag}_{ss}_err-{ee}"
            sens_dict[tag] = copy.deepcopy(sens_array)

            if err_chain_dict[ee] is not None:
                err_int_opts = sens.ErrIntOpts()
                sens_dict[tag].set_error_chain(err_chain_dict[ee],
                                               err_int_opts)

    return sens_dict


#-------------------------------------------------------------------------------
def sens_arrays_2d_dict() -> dict[str,sens.SensorsPoint]:
    sim_data = pointsensmech.simdata_mech_2d()
    sens_data_dict = pointsensmech.sens_data_2d_dict(sim_data)
    tag = "vec2d"
    return gen_sens_arrays_2d_dict(sim_data,sens_data_dict,tag)

def sens_arrays_2d_analytic_dict() -> dict[str,sens.SensorsPoint]:
    sim_data = simdata_vec_2d_analytic()
    sens_data_dict = pointsensmech.sens_data_2d_dict(sim_data)
    tag = "vec2d_analytic"
    return gen_sens_arrays_2d_dict(sim_data,sens_data_dict,tag)

def sens_arrays_2d_analytic_nomesh_dict() -> dict[str,sens.SensorsPoint]:
    sim_data = simdata_vec_2d_analytic_nomesh()
    sens_data_dict = pointsensmech.sens_data_2d_dict(sim_data)
    tag = "vec2d_analytic_nomesh"
    return gen_sens_arrays_2d_dict(sim_data,sens_data_dict,tag)


def sens_arrays_3d_dict() -> dict[str,sens.SensorsPoint]:
    sim_data = pointsensmech.simdata_mech_3d()
    sens_data_dict = pointsensmech.sens_data_3d_dict(sim_data)
    tag = "vec3d"
    return gen_sens_arrays_3d_dict(sim_data,sens_data_dict,tag)


def sens_arrays_3d_nomesh_dict() -> dict[str,sens.SensorsPoint]:
    sim_data = pointsensmech.simdata_mech_3d_nomesh()
    sens_data_dict = pointsensmech.sens_data_3d_dict(sim_data)
    tag = "vec3d_nomesh"
    return gen_sens_arrays_3d_dict(sim_data,sens_data_dict,tag)
