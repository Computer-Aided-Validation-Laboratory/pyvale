# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np
from pyvale.sensorsim.sensordata import SensorData
from pyvale.sensorsim.errorsimulator import (IErrSimulator,
                                         EErrType,
                                         EErrDep)
from pyvale.sensorsim.generatorsrandom import IGenRandom


class ErrRandGen(IErrSimulator):
    """Sensor random error calculator based on sampling a user specified random
    number generator implementing the `IGeneratorRandom` interface.

    Implements the `IErrSimulator` interface.
    """
    __slots__ = ("_generator","_err_dep")

    def __init__(self,
                 generator: IGenRandom,
                 err_dep: EErrDep = EErrDep.INDEPENDENT) -> None:
        """
        Parameters
        ----------
        generator : IGeneratorRandom
            Interface for a user specified random number generator.
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
        return EErrType.RANDOM

    def reseed(self, seed: int | None = None) -> None:        
        self._generator.reseed(seed)

    def sim_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:
        rand_errs = self._generator.generate(shape=err_basis.shape)

        return (rand_errs,sens_data)


class ErrRandGenPercent(IErrSimulator):
    """Random error calculator based on sampling a user specified random
    number generator implementing the `IGeneratorRandom` interface. This class
    assumes the random generator is for a percentage error based on the input
    error basis and therefore it supports error dependence.

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
        generator : IGeneratorRandom
            Interface for a user specified random number generator.
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
        return EErrType.RANDOM

    def reseed(self, seed: int | None = None) -> None:
        self._generator.reseed(seed)

    def sim_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:
        rand_errs = err_basis \
            * self._generator.generate(shape=err_basis.shape)/100.0

        return (rand_errs,sens_data)
