#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================
from dataclasses import dataclass
import pytest
import numpy as np
import pyvale

@dataclass(slots=True)
class CheckData:
    seed: int = 0
    sensor_data: pyvale.SensorData | None = None
    err_basis: np.ndarray | None = None
    meas_shape: tuple[int,int,int] = (6,2,10)

    def __post_init__(self) -> None:
        self.sensor_data = pyvale.SensorData()
        self.err_basis = 5.0*np.ones(self.meas_shape)


@pytest.fixture()
def check_data() -> CheckData:
    return CheckData()


def check_err_calc_rand(err_calc: pyvale.IErrCalculator,
                        data: CheckData) -> np.ndarray:

    err_calc.set_error_dep(pyvale.EErrDep.DEPENDENT)
    assert err_calc.get_error_dep() == pyvale.EErrDep.DEPENDENT

    err_calc.set_error_dep(pyvale.EErrDep.INDEPENDENT)
    assert err_calc.get_error_dep() == pyvale.EErrDep.INDEPENDENT

    assert err_calc.get_error_type() == pyvale.EErrType.RANDOM

    (err_mat,_) = err_calc.calc_errs(err_basis=data.err_basis,
                                     sens_data=data.sensor_data)
    assert err_mat.shape == data.meas_shape

    return err_mat


def test_ErrRandUniform(check_data: CheckData) -> None:
    low = 1.0
    high = 1.0
    err_calc = pyvale.ErrRandUnif(low=low,
                                     high=high,
                                     err_dep=pyvale.EErrDep.INDEPENDENT,
                                     seed=check_data.seed)

    err_mat = check_err_calc_rand(err_calc,check_data)

    check_mat = np.random.default_rng(check_data.seed) \
        .uniform(low,high,check_data.meas_shape)

    assert np.allclose(err_mat,check_mat)


def test_ErrRandUnifPercent(check_data: CheckData) -> None:
    low_percent = -1.0
    high_percent = 1.0
    err_calc = pyvale.ErrRandUnifPercent(low_percent=low_percent,
                                         high_percent=high_percent,
                                         err_dep=pyvale.EErrDep.INDEPENDENT,
                                         seed=check_data.seed)

    err_mat = check_err_calc_rand(err_calc,check_data)

    check_mat = np.random.default_rng(check_data.seed) \
                .uniform(low_percent/100,high_percent/100,check_data.meas_shape)
    check_mat = check_data.err_basis * check_mat
    assert np.allclose(err_mat,check_mat)


def test_ErrRandNormal(check_data: CheckData) -> None:
    std = 5.0
    err_calc = pyvale.ErrRandNorm(std=std,
                                    err_dep=pyvale.EErrDep.INDEPENDENT,
                                    seed=check_data.seed)

    err_mat = check_err_calc_rand(err_calc,check_data)

    check_mat = np.random.default_rng(check_data.seed) \
        .normal(loc=0.0,scale=std,size=check_data.meas_shape)

    assert np.allclose(err_mat,check_mat)


def test_ErrRandNormPercent(check_data: CheckData) -> None:
    std_percent = 5.0
    err_calc = pyvale.ErrRandNormPercent(std_percent=std_percent,
                                    err_dep=pyvale.EErrDep.INDEPENDENT,
                                    seed=check_data.seed)

    err_mat = check_err_calc_rand(err_calc,check_data)

    check_mat = np.random.default_rng(check_data.seed) \
        .normal(loc=0.0,scale=std_percent/100,size=check_data.meas_shape)
    check_mat = check_data.err_basis * check_mat
    assert np.allclose(err_mat,check_mat)


def test_ErrRandGenerator(check_data: CheckData) -> None:
    std = 5.0

    rand_gen = pyvale.GenNormal(std=std,mean=0.0,seed=check_data.seed)

    err_calc = pyvale.ErrRandGenerator(rand_gen)

    err_mat = check_err_calc_rand(err_calc,check_data)

    check_mat = np.random.default_rng(check_data.seed) \
        .normal(loc=0.0,scale=std,size=check_data.meas_shape)

    assert np.allclose(err_mat,check_mat)



