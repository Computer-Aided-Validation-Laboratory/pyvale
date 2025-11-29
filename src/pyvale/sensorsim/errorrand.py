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
        """Gets the error dependence state for this error calculator. An
        independent error is calculated based on the input truth values as the
        error basis. A dependent error is calculated based on the accumulated
        sensor reading from all preceeding errors in the chain.

        For this class errors are calculated independently regardless.

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

        For this class errors are calculated independently regardless.

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
        return EErrType.RANDOM

    def reseed(self, seed: int | None = None) -> None:
        """Reseeds the random generators of the error simulator. Mainly used for
        multi-processed simulations which inherit the same seed as the main 
        process so need to be reseeded.
        
        Parameters
        ----------
        seed : int | None, optional
            Integer seed for the random number generator, by default None. If 
            None then the seed is generated using OS entropy (see numpy docs).
        """
        
        self._generator.reseed(seed)

    def sim_errs(self,
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
        return EErrType.RANDOM

    def reseed(self, seed: int | None = None) -> None:
        """Reseeds the random generators of the error simulator. Mainly used for
        multi-processed simulations which inherit the same seed as the main 
        process so need to be reseeded.

        Parameters
        ----------
        seed : int | None, optional
            Integer seed for the random number generator, by default None. If 
            None then the seed is generated using OS entropy (see numpy docs).
        """

        self._generator.reseed(seed)

    def sim_errs(self,
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
        rand_errs = err_basis \
            * self._generator.generate(shape=err_basis.shape)/100.0

        return (rand_errs,sens_data)
