#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================
from dataclasses import dataclass
import pytest
import numpy as np
import mooseherder as mh
import pyvale as pyv


#-------------------------------------------------------------------------------
# NOTE
# - Need to produce 'gold' output
# - Need analytic test cases

# TODO
# - different sensor positions
# - sample times aligned to sim or not
# - 2D or 2D simulation
# - get vs calc methods
# - Truth:
#   - Area averaging
# - Errors:
#   - No errors
#   - Statistical errors with fixed seeds
#   - Field errors
#   - Dependence

# 1) No errors, pure interpolation in 2D/3D
# 2)

#-------------------------------------------------------------------------------
# Test Resources

@pytest.fixture
def simdata_2d() -> mh.SimData:

    data_path = pyv.DataSet.thermal_2d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=None)

    return sim_data

@pytest.fixture
def simdata_3d() -> mh.SimData:

    data_path = pyv.DataSet.thermal_3d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=None)

    return sim_data


# @pytest.fixture(autouse=True)
# def setup_teardown(dir_manager):
#     # Before test prep
#     yield
#     # Post test clean up
#     dir_manager.clear_dirs()

#-------------------------------------------------------------------------------
# Tests
def test_2d_noerrs() -> None:
    pass
