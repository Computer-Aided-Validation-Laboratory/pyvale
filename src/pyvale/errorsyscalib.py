# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from typing import Callable
import numpy as np
from pyvale.errorcalculator import (IErrCalculator,
                                         EErrType,
                                         EErrDep)
from pyvale.sensordata import SensorData

# TODO: add option to use Newton's method for function inversion instead of a
# cal table.

class ErrSysCalibration(IErrCalculator):
    """Systematic error calculator for calibration errors. The user specifies an
    assumed calibration and a ground truth calibration function. The ground
    truth calibration function is inverted and linearly interpolated numerically
    based on the number of divisions specified by the user.

    Implements the `IErrCalculator` interface.
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
        """Gets the error dependence state for this error calculator. An
        independent error is calculated based on the input truth values as the
        error basis. A dependent error is calculated based on the accumulated
        sensor reading from all preceeding errors in the chain.

        Returns
        -------
        EErrDependence
            Enumeration defining INDEPENDENT or DEPENDENT behaviour.
        """
        return self._err_dep

    def set_error_dep(self, dependence: EErrDep) -> None:
        """Sets the error dependence state for this error calculator. An
        independent error is calculated based on the input truth values as the
        error basis. A dependent error is calculated based on the accumulated
        sensor reading from all preceeding errors in the chain.

        Parameters
        ----------
        dependence : EErrDependence
            Enumeration defining INDEPENDENT or DEPENDENT behaviour.
        """
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        """Gets the error type.

        Returns
        -------
        EErrType
            Enumeration definining RANDOM or SYSTEMATIC error types.
        """
        return EErrType.SYSTEMATIC

    def calc_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:
        """Calculates the error array based on the size of the input.

        Parameters
        ----------
        err_basis : np.ndarray
            Array of values with the same dimensions as the sensor measurement
            matrix.
        sens_data : SensorData
            The accumulated sensor state data for all errors prior to this one.

        Returns
        -------
        tuple[np.ndarray, SensorData]
            Tuple containing the calculated error array and pass through of the
            sensor data object as it is not modified by this class. The returned
            error array has the same shape as the input error basis.
        """
        # shape=(n_sens,n_comps,n_time_steps)
        signal_from_field = np.interp(err_basis,
                                    self._truth_cal_table[:,1],
                                    self._truth_cal_table[:,0])
        # shape=(n_sens,n_comps,n_time_steps)
        field_from_assumed_calib = self._assumed_calib(signal_from_field)

        sys_errs = field_from_assumed_calib - err_basis

        return (sys_errs,sens_data)

