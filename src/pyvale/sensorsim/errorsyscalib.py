# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from typing import Callable
import numpy as np
from pyvale.sensorsim.errorsimulator import (IErrSimulator,
                                         EErrType,
                                         EErrDep)
from pyvale.sensorsim.sensordata import SensorData

# TODO: add option to use Newton's method for function inversion instead of a
# cal table.

class ErrSysCalibration(IErrSimulator):
    """Systematic error calculator for calibration errors. The user specifies an
    assumed calibration and a ground truth calibration function. The ground
    truth calibration function is inverted and linearly interpolated numerically
    based on the number of divisions specified by the user.

    Implements the `IErrSimulator` interface.
    """
    __slots__ = ("_assumed_cali","_truth_calib","_cal_range","_n_cal_divs",
                 "_err_dep","_truth_calc_table")

    def __init__(self,
                 assumed_calib: Callable[[np.ndarray],np.ndarray],
                 truth_calib: Callable[[np.ndarray],np.ndarray],
                 cal_range: tuple[float,float],
                 n_cal_divs: int = 10000,
                 err_dep: EErrDep = EErrDep.INDEPENDENT) -> None:
        """
        Parameters
        ----------
        assumed_calib : Callable[[np.ndarray],np.ndarray]
            Assumed calibration function taking the input unitless 'signal' and
            converting it to the same units as the physical field being sampled
            by the sensor array.
        truth_calib : Callable[[np.ndarray],np.ndarray]
            Assumed calibration function taking the input unitless 'signal' and
            converting it to the same units as the physical field being sampled
            by the sensor array.
        cal_range : tuple[float,float]
            Range over which the calibration functions are valid. This is
            normally based on a voltage range such as (0,10) volts.
        n_cal_divs : int, optional
            Number of divisions to discretise the the truth calibration function
            for numerical inversion, by default 10000.
        err_dep : EErrDependence, optional
            Error calculation dependence, by default EErrDependence.INDEPENDENT.
        """
        self._assumed_calib = assumed_calib
        self._truth_calib = truth_calib
        self._cal_range = cal_range
        self._n_cal_divs = n_cal_divs
        self._err_dep = err_dep

        self._truth_cal_table = np.zeros((n_cal_divs,2))
        self._truth_cal_table[:,0] = np.linspace(cal_range[0],
                                                cal_range[1],
                                                n_cal_divs)
        self._truth_cal_table[:,1] = self._truth_calib(
                                        self._truth_cal_table[:,0])

    def get_error_dep(self) -> EErrDep:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDep) -> None:
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def reseed(self, seed: int | None = None) -> None:
        pass

    def sim_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:
        # shape=(n_sens,n_comps,n_time_steps)
        signal_from_field = np.interp(err_basis,
                                    self._truth_cal_table[:,1],
                                    self._truth_cal_table[:,0])
        # shape=(n_sens,n_comps,n_time_steps)
        field_from_assumed_calib = self._assumed_calib(signal_from_field)

        sys_errs = field_from_assumed_calib - err_basis

        return (sys_errs,sens_data)

