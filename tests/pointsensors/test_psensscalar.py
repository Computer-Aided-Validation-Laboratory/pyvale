#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================
import pytest
import numpy as np
import mooseherder as mh
import pyvale as pyv
import pyvale.verif.psensconst as psensconst
import pyvale.verif.psensscalar as psensscalar


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

# @pytest.fixture
# def simdata_2d() -> mh.SimData:

#     data_path = pyv.DataSet.thermal_2d_path()
#     sim_data = mh.ExodusReader(data_path).read_all_sim_data()
#     sim_data = pyv.scale_length_units(scale=1000.0,
#                                       sim_data=sim_data,
#                                       disp_comps=None)

#     return sim_data

# @pytest.fixture
# def simdata_3d() -> mh.SimData:

#     data_path = pyv.DataSet.thermal_3d_path()
#     sim_data = mh.ExodusReader(data_path).read_all_sim_data()
#     sim_data = pyv.scale_length_units(scale=1000.0,
#                                       sim_data=sim_data,
#                                       disp_comps=None)

#     return sim_data

# @pytest.fixture(autouse=True)
# def setup_teardown(dir_manager):
#     # Before test prep
#     yield
#     # Post test clean up
#     dir_manager.clear_dirs()

#-------------------------------------------------------------------------------
# Tests
def test_gold_scalar2d() -> None:
    """Gold regression testing for all scalar field point sensors in 2D.
    """
    sens_dict = psensscalar.sens_2d_dict()

    fails = []
    for ss in sens_dict:
        measurements = sens_dict[ss].calc_measurements()

        load_path = psensconst.GOLD_PATH / f"{ss}.npy"
        if load_path.is_file():
            gold = np.load(load_path)

            if not np.allclose(measurements,gold):
                fails.append(f"Gold check failed for: {ss}")
        else:
            fails.append(f"Gold file does not exist for: {ss}")

    assert not fails, "\n".join(fails)


def test_get_meas_scalar2d() -> None:
    """Tests that get does not resample from probability distributions.
    """
    sens_dict = psensscalar.sens_2d_dict()

    fails = []
    for ss in sens_dict:
        calc_meas = sens_dict[ss].calc_measurements()
        get_meas = sens_dict[ss].get_measurements()

        if not np.allclose(calc_meas, get_meas):
            fails.append(f"Get does not equal calc for: {ss}")

    assert not fails, "\n".join(fails)

# TODO: check that for the last time step of all sensors the measurement is not zero
# if it is zero we are interpolating outside the mesh


