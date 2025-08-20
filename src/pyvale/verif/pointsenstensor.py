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
applied to tensor fields.
"""

# TODO
# - Calibration errors for tensor fields

def simdata_tens_2d_analytic() -> mh.SimData:
    (sim_data,_) = asd.tensor_linear_2d()
    return sim_data

def simdata_tens_2d_analytic_nomesh() -> mh.SimData:
    sim_data = simdata_tens_2d_analytic()
    sim_data.connect = None
    return sim_data

def sens_array_2d_noerrs(sim_data: mh.SimData,
                   sens_data: sens.SensorData) -> sens.SensorArrayPoint:
    descriptor = sens.SensorDescriptorFactory.strain_descriptor()
    field_name = "strain"
    norm_comps = ("strain_xx","strain_yy")
    dev_comps = ("strain_xy",)
    field = sens.FieldTensor(sim_data,
                            field_name=field_name,
                            norm_comps=norm_comps,
                            dev_comps=dev_comps,
                            elem_dims=2)
    sens_array = sens.SensorArrayPoint(sens_data,
                                      field,
                                      descriptor)
    return sens_array


def sens_array_3d_noerrs(sim_data: mh.SimData,
                   sens_data: sens.SensorData) -> sens.SensorArrayPoint:
    descriptor = sens.SensorDescriptorFactory.strain_descriptor()
    field_name = "strain"
    norm_comps = ("strain_xx","strain_yy","strain_zz")
    dev_comps = ("strain_xy","strain_yz","strain_xz")
    field = sens.FieldTensor(sim_data,
                            field_name=field_name,
                            norm_comps=norm_comps,
                            dev_comps=dev_comps,
                            elem_dims=3)
    sens_array =  sens.SensorArrayPoint(sens_data,
                                       field,
                                       descriptor)
    return sens_array

#-------------------------------------------------------------------------------
def gen_sens_arrays_2d_dict(sim_data: mh.SimData,
                            sens_data_dict: dict[str,sens.SensorData],
                            sens_tag: str,
                            ) -> dict[str,sens.SensorArrayPoint]:
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
                err_int = sens.ErrIntegrator(err_chain_dict[ee],
                                            sens_data_dict[ss],
                                            sens_dict[tag].get_measurement_shape(),
                                            err_int_opts=err_int_opts)
                sens_dict[tag].set_error_integrator(err_int)

    return sens_dict

def gen_sens_arrays_3d_dict(sim_data: mh.SimData,
                            sens_data_dict: dict[str,sens.SensorData],
                            sens_tag: str,
                            ) -> dict[str,sens.SensorArrayPoint]:
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
                err_int = sens.ErrIntegrator(err_chain_dict[ee],
                                            sens_data_dict[ss],
                                            sens_dict[tag].get_measurement_shape(),
                                            err_int_opts=err_int_opts)
                sens_dict[tag].set_error_integrator(err_int)

    return sens_dict
#-------------------------------------------------------------------------------
def sens_arrays_2d_dict() -> dict[str,sens.SensorArrayPoint]:
    sim_data = pointsensmech.simdata_mech_2d()
    sens_data_dict = pointsensmech.sens_data_2d_dict(sim_data)
    tag = "tens2d"
    return gen_sens_arrays_2d_dict(sim_data,sens_data_dict,tag)

def sens_arrays_2d_analytic_dict() -> dict[str,sens.SensorArrayPoint]:
    sim_data = simdata_tens_2d_analytic()
    sens_data_dict = pointsensmech.sens_data_2d_dict(sim_data)
    tag = "tens2d_analytic"
    return gen_sens_arrays_2d_dict(sim_data,sens_data_dict,tag)

def sens_arrays_2d_analytic_nomesh_dict() -> dict[str,sens.SensorArrayPoint]:
    sim_data = simdata_tens_2d_analytic_nomesh()
    sens_data_dict = pointsensmech.sens_data_2d_dict(sim_data)
    tag = "tens2d_analytic_nomesh"
    return gen_sens_arrays_2d_dict(sim_data,sens_data_dict,tag)


def sens_arrays_3d_dict() -> dict[str,sens.SensorArrayPoint]:
    sim_data = pointsensmech.simdata_mech_3d()
    sens_data_dict = pointsensmech.sens_data_3d_dict(sim_data)
    tag = "tens3d"
    return gen_sens_arrays_3d_dict(sim_data,sens_data_dict,tag)

def sens_arrays_3d_nomesh_dict() -> dict[str,sens.SensorArrayPoint]:
    sim_data = pointsensmech.simdata_mech_3d_nomesh()
    sens_data_dict = pointsensmech.sens_data_3d_dict(sim_data)
    tag = "tens3d_nomesh"
    return gen_sens_arrays_3d_dict(sim_data,sens_data_dict,tag)
