#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================
import copy
import numpy as np
import mooseherder as mh
import pyvale as pyv
import pyvale.verif.psens as psens
import pyvale.verif.psensconst as psensconst

"""
DEVELOPER VERIFICATION MODULE
--------------------------------------------------------------------------------
This module contains developer utility functions used for verification testing
of the point sensor simulation toolbox in pyvale.

Specifically, this module contains functions used for testing point sensors
applied to scalar field.
"""

def simdata_2d() -> mh.SimData:
    data_path = pyv.DataSet.mechanical_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=("disp_x","disp_y"))
    return sim_data


def simdata_3d() -> mh.SimData:
    data_path = pyv.DataSet.element_case_path(pyv.EElemTest.HEX20)
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    field_comps = ("disp_x","disp_y","disp_z")
    sim_data = pyv.scale_length_units(scale=1000.0,
                                        sim_data=sim_data,
                                        disp_comps=field_comps)
    return sim_data

def sens_pos_2d() -> dict[str,np.ndarray]:
    sim_dims = pyv.get_sim_dims(simdata_2d())
    sens_pos = {}

    x_lims = sim_dims["x"]
    y_lims = sim_dims["y"]
    z_lims = (0,0)

    n_sens = (1,4,1)
    sens_pos["line-4"] = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    n_sens = (2,3,1)
    sens_pos["grid-23"] = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    return sens_pos


def sens_pos_3d() -> dict[str,np.ndarray]:
    sens_pos = {}
    sens_pos["cent-cube"] = np.array(((5.0,0.0,5.0),
                                      (5.0,10.0,5.0),
                                      (5.0,5.0,0.0),
                                      (5.0,5.0,10.0),
                                      (0.0,5.0,5.0),
                                      (10.0,5.0,5.0),))
    return sens_pos
