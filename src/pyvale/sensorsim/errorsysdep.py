# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import enum
from typing import Callable
import numpy as np
from pyvale.sensorsim.sensordata import SensorData
from pyvale.sensorsim.errorsimulator import (IErrSimulator,
                                         EErrType,
                                         EErrDep)


class ERoundMethod(enum.Enum):
    """Enumeration used to specify the method for rounding floats to integers.
    """
    ROUND = enum.auto()
    FLOOR = enum.auto()
    CEIL = enum.auto()


def _select_round_method(method: ERoundMethod) -> Callable:
    """Helper function for selecting the rounding method based on the user
    specified enumeration. Returns a numpy function for rounding.

    Parameters
    ----------
    method : ERoundMethod
        Enumeration specifying the rounding method.

    Returns
    -------
    Callable
        numpy rounding method as np.floor, np.ceil or np.round.
    """
    if method == ERoundMethod.FLOOR:
        return np.floor
    if method == ERoundMethod.CEIL:
        return np.ceil
    return np.round


class ErrSysRoundOff(IErrSimulator):
    """Systematic error calculator for round off error. The user can specify the
    floor, ceiling or nearest integer method for rounding. The user can also
    specify a base to round to that defaults 1. Implements the `IErrSimulator`
    interface.
    """
    __slots__ = ("_base","_method","_err_dep")

    def __init__(self,
                 method: ERoundMethod = ERoundMethod.ROUND,
                 base: float = 1.0,
                 err_dep: EErrDep = EErrDep.DEPENDENT) -> None:
        """
        Parameters
        ----------
        method : ERoundMethod, optional
            Enumeration specifying the rounding method, by default
            ERoundMethod.ROUND.
        base : float, optional
            Base to round to, by default 1.0.
        err_dep : EErrDependence, optional
            Error calculation dependence, by default EErrDependence.DEPENDENT.
        """
        self._base = base
        self._method = _select_round_method(method)
        self._err_dep = err_dep

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
        rounded_measurements = self._base*self._method(err_basis/self._base)

        return (rounded_measurements - err_basis,sens_data)


class ErrSysDigitisation(IErrSimulator):
    """Systematic error calculator for digitisation error base on a user
    specified number of bits per physical unit and rounding method. Implements
    the `IErrSimulator` interface.
    """
    __slots__ = ("_units_per_bit","_method","_err_dep")

    def __init__(self,
                 bits_per_unit: float,
                 method: ERoundMethod = ERoundMethod.ROUND,
                 err_dep: EErrDep = EErrDep.DEPENDENT) -> None:
        """
        Parameters
        ----------
        bits_per_unit : float
            The number of bits per physical unit used to determine the
            digitisation error.
        method : ERoundMethod, optional
            User specified rounding method, by default ERoundMethod.ROUND.
        err_dep : EErrDependence, optional
            Error calculation dependence, by default EErrDependence.DEPENDENT.
        """
        self._units_per_bit = 1/float(bits_per_unit)
        self._method = _select_round_method(method)
        self._err_dep = err_dep

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
        rounded_measurements = self._units_per_bit*self._method(
            err_basis/self._units_per_bit)

        return (rounded_measurements - err_basis,sens_data)


class ErrSysSaturation(IErrSimulator):
    """Systematic error calculator for saturation error base on user specified
    minimum and maximum measurement values. Implements the `IErrSimulator`
    interface.

    NOTE: For this error to function as expected and clamp the measurement
    within the specified range it must be placed last in the error chain and
    the behaviour must be set to: EErrDependence.DEPENDENT.
    """
    __slots__ = ("_min","_max","_err_dep")

    def __init__(self,
                 meas_min: float,
                 meas_max: float) -> None:
        """
        Parameters
        ----------
        meas_min : float
            Minimum value to saturate the measurement to.
        meas_max : float
            Maximum value to saturate the measurement to.

        Raises
        ------
        ValueError
            Raised if the user specified minimum measurement is greater than the
            maximum measurement.
        """
        if meas_min > meas_max:
            raise ValueError("Minimum must be smaller than maximum for "+
                             "systematic error saturation")

        self._min = meas_min
        self._max = meas_max
        self._err_dep = EErrDep.DEPENDENT

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
        saturated = np.copy(err_basis)
        saturated[saturated > self._max] = self._max
        saturated[saturated < self._min] = self._min

        return (saturated - err_basis,sens_data)
