# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np
from pyvale.sensorsim.errorsimulator import (IErrSimulator,
                                         EErrType,
                                         EErrDep)
from pyvale.sensorsim.generatorsrandom import IGenRandom
from pyvale.sensorsim.sensordata import SensorData


class ErrSysOffset(IErrSimulator):
    """Systematic error calculator applying a constant offset to all simulated
    sensor measurements. Implements the `IErrSimulator` interface.
    """
    __slots__ = ("_offset","_err_dep")

    def __init__(self,
                 offset: float,
                 err_dep: EErrDep = EErrDep.INDEPENDENT) -> None:
        """
        Parameters
        ----------
        offset : float
            Constant offset to apply to all simulated measurements from the
            sensor array.
        err_dep : EErrDependence, optional
            Error calculation dependence, by default EErrDependence.INDEPENDENT.
        """
        self._offset = offset
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
        return (self._offset*np.ones(shape=err_basis.shape),sens_data)


class ErrSysOffsetPercent(IErrSimulator):
    """Systematic error calculator applying a constant offset as a percentage of
    the sensor reading to each individual simulated sensor measurement.
    Implements the `IErrSimulator` interface.
    """
    __slots__ = ("_offset_percent","_err_dep")

    def __init__(self,
                 offset_percent: float,
                 err_dep: EErrDep = EErrDep.INDEPENDENT) -> None:
        """
        Parameters
        ----------
        offset_percent : float
            Percentage offset to apply to apply to all simulated measurements
            from the sensor array.
        err_dep : EErrDependence, optional
            Error calculation dependence, by default EErrDependence.INDEPENDENT
        """
        self._offset_percent = offset_percent
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
        return (self._offset_percent/100 *
                err_basis *
                np.ones(shape=err_basis.shape),
                sens_data)

class ErrSysGen(IErrSimulator):
    """Systematic error calculator for applying a unique offset to each sensor
    by sample from a user specified probability distribution (an implementation
    of the `IGeneratorRandom` interface).

    Implements the `IErrSimulator` interface.
    """
    __slots__ = ("_generator","_err_dep")

    def __init__(self,
                 generator: IGenRandom,
                 err_dep: EErrDep = EErrDep.INDEPENDENT) -> None:
        """
        Parameters
        ----------
        generator : IGenRandom
            Random generator object used to calculate the systematic error in
            simulation units.
        err_dep : EErrDependence, optional
            Error calculation dependence, by default EErrDependence.INDEPENDENT.
        """
        self._generator = generator
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDep:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDep) -> None:
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def reseed(self, seed: int | None = None) -> None:
        self._generator.reseed(seed)

    def sim_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:
        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1

        sys_errs = self._generator.generate(shape=err_shape)

        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))

        return (sys_errs,sens_data)


class ErrSysGenPercent(IErrSimulator):
    """Systematic error calculator for applying a unique percentage offset to
    each sensor by sample from a user specified probability distribution (an
    implementation of the `IGeneratorRandom` interface). This class assumes the
    random generator is for a percentage error based on the input error basis
    and therefore it supports error dependence.

    The percentage error is calculated based on the ground truth if the error
    dependence is `INDEPENDENT` or based on the accumulated sensor measurement
    if the dependence is `DEPENDENT`.

    Implements the `IErrSimulator` interface.
    """
    __slots__ = ("_generator","_err_dep")

    def __init__(self,
                 generator: IGenRandom,
                 err_dep: EErrDep = EErrDep.INDEPENDENT) -> None:
        """
        Parameters
        ----------
        generator : IGenRandom
            Random generator which returns a percentage error in the range
            (0,100)
        err_dep : EErrDep, optional
            Error calculation dependence, by default EErrDep.INDEPENDENT
        """
        self._generator = generator
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDep:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDep) -> None:
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def reseed(self, seed: int | None = None) -> None:
        self._generator.reseed(seed)

    def sim_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:
        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1

        sys_errs = self._generator.generate(shape=err_shape)
        # Convert percent to decimal
        sys_errs = sys_errs/100.0

        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))
        sys_errs = err_basis * sys_errs

        return (sys_errs,sens_data)

