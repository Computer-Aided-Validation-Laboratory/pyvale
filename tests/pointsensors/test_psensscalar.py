#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================
import pytest
import numpy as np
import pyvale.verif.psens as psens
import pyvale.verif.psensscalar as psensscalar


#-------------------------------------------------------------------------------
# TODO: Tests
# - Gold for vector and tensor fields
# - Analytic test cases
# - Gold for multi-physics experiments
# - Logic for vector and tensor rotations
# - Area averaging for sensors on different faces in 2D

# VECTOR/TENSOR FIELDS:
# - rotation of area averaging

#-------------------------------------------------------------------------------

def test_gold_scalar2d() -> None:
    """Gold regression testing for all scalar field point sensors in 2D.
    """
    fails = psens.check_gold(psensscalar.sens_2d_dict())
    assert not fails, "\n".join(fails)


def test_gold_scalar3d() -> None:
    """Gold regression testing for all scalar field point sensors in 3D.
    """
    fails = psens.check_gold(psensscalar.sens_3d_dict())
    assert not fails, "\n".join(fails)


def test_get_meas_scalar() -> None:
    """Tests that get does not resample from probability distributions.
    """
    fails = []

    sens_dict = psensscalar.sens_2d_dict()

    for ss in sens_dict:
        calc_meas = sens_dict[ss].calc_measurements()
        get_meas = sens_dict[ss].get_measurements()

        if not np.allclose(calc_meas, get_meas):
            fails.append(f"2D, get does not equal calc for: {ss}")

    sens_dict = psensscalar.sens_3d_dict()
    for ss in sens_dict:
        calc_meas = sens_dict[ss].calc_measurements()
        get_meas = sens_dict[ss].get_measurements()

        if not np.allclose(calc_meas, get_meas):
            fails.append(f"3D, get does not equal calc for: {ss}")

    assert not fails, "\n".join(fails)

# TODO: check that for the last time step of all sensors the measurement is not zero
# if it is zero we are interpolating outside the mesh


