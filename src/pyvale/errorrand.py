# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np
from pyvale.sensordata import SensorData
from pyvale.errorcalculator import (IErrCalculator,
                                         EErrType,
                                         EErrDep)
from pyvale.generatorsrandom import IGenRandom


class ErrRandUnif(IErrCalculator):
    """Random error calculator based on uniform sampling of an interval
    specified by its upper and lower bound.

    Implements the `IErrCalculator` interface.
    """
    __slots__ = ("low","high","rng","err_dep")

    def __init__(self,
                 low: float,
                 high: float,
                 err_dep: EErrDep = EErrDep.INDEPENDENT,
                 seed: int | None = None) -> None:
        """
        Parameters
        ----------
        low : float
            Lower bound of the uniform random generator.
        high : float
            Upper bound of the uniform random generator.
        err_dep : EErrDependence, optional
            Error calculation dependence, by default EErrDependence.INDEPENDENT.
        seed : int | None, optional
            Optional seed for the random generator to allow for replicable
            behaviour, by default None.

        Raises
        ------
        ValueError
            Raised if the user specified a lower bound that is higher than the
            upper bound.
        """

        if low > high:
            raise ValueError("The upper bound must be higher than the lower "
                              + "bound for this error.")
        self.low = low
        self.high = high
        self.rng = np.random.default_rng(seed)
        self.err_dep = err_dep

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
        return self.err_dep

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
        self.err_dep = dependence

    def get_error_type(self) -> EErrType:
        """Gets the error type.

        Returns
        -------
        EErrType
            Enumeration definining RANDOM or SYSTEMATIC error types.
        """
        return EErrType.RANDOM

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
        rand_errs = self.rng.uniform(low=self.low,
                                     high=self.high,
                                     size=err_basis.shape)

        return (rand_errs,sens_data)


class ErrRandUnifPercent(IErrCalculator):
    """Random error calculator based on a percentage error based on sampling
    from a uniform probability distribution specified by its upper and lower
    bound (in percent).

    The percentage error is calculated based on the ground truth if the error
    dependence is `INDEPENDENT` or based on the accumulated sensor measurement
    if the dependence is `DEPENDENT`.

    Implements the `IErrCalculator` interface.
    """
    __slots__ = ("low","high","rng","err_dep")

    def __init__(self,
                 low_percent: float,
                 high_percent: float,
                 err_dep: EErrDep = EErrDep.INDEPENDENT,
                 seed: int | None = None) -> None:
        """
        Parameters
        ----------
        low_percent : float
            Lower percentage bound of the uniform random generator.
        high_percent : float
            Upper percentage bound of the uniform random generator.
        err_dep : EErrDependence, optional
            Error calculation dependence, by default EErrDependence.INDEPENDENT.
        seed : int | None, optional
            Optional seed for the random generator to allow for replicable
            behaviour, by default None.

        Raises
        ------
        ValueError
            Raised if the user specified a lower bound that is higher than the
            upper bound.
        """

        if low_percent > high_percent:
            raise ValueError("The upper percentage bound must be higher than "
                              + "the lower percentage bound for this error.")

        self.low = low_percent/100
        self.high = high_percent/100
        self.rng = np.random.default_rng(seed)
        self.err_dep = err_dep

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
        return self.err_dep

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
        self.err_dep = dependence

    def get_error_type(self) -> EErrType:
        """Gets the error type.

        Returns
        -------
        EErrType
            Enumeration definining RANDOM or SYSTEMATIC error types.
        """
        return EErrType.RANDOM

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
        norm_rand = self.rng.uniform(low=self.low,
                                    high=self.high,
                                    size=err_basis.shape)

        return (err_basis*norm_rand,sens_data)


class ErrRandNorm(IErrCalculator):
    """Random error calculator based on sampling of a normal (Gaussian)
    distribution specified using the standard deviation with an assumed zero
    mean. A non-zero mean is a systematic error and should be specified using
    `ErrSysOffset`.

    Implements the `IErrCalculator` interface.
    """
    __slots__ = ("std","rng","err_dep")

    def __init__(self,
                 std: float,
                 err_dep: EErrDep = EErrDep.INDEPENDENT,
                 seed: int | None = None) -> None:
        """
        Parameters
        ----------
        std : float
            Standard deviation of the normal distribution to sample from.
        err_dep : EErrDependence, optional
            Error calculation dependence, by default EErrDependence.INDEPENDENT.
        seed : int | None, optional
            Optional seed for the random generator to allow for replicable
            behaviour, by default None.
        """
        self.std = np.abs(std)
        self.rng = np.random.default_rng(seed)
        self.err_dep = err_dep

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
        return self.err_dep

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
        self.err_dep = dependence

    def get_error_type(self) -> EErrType:
        """Gets the error type.

        Returns
        -------
        EErrType
            Enumeration definining RANDOM or SYSTEMATIC error types.
        """
        return EErrType.RANDOM

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
        rand_errs = self.rng.normal(loc=0.0,
                                    scale=self.std,
                                    size=err_basis.shape)

        return (rand_errs,sens_data)


class ErrRandNormPercent(IErrCalculator):
    """Sensor random error calculator based on sampling of a normal (Gaussian)
    distribution specified using the standard deviation with an assumed zero
    mean. This error is calculated as a percentage of the input error basis.
    Note that a non-zero mean is a systematic error and should be specified
    using `ErrSysOffset`.

    The percentage error is calculated based on the ground truth if the error
    dependence is `INDEPENDENT` or based on the accumulated sensor measurement
    if the dependence is `DEPENDENT`.

    Implements the `IErrCalculator` interface.
    """
    __slots__ = ("_std","_rng","_err_dep")

    def __init__(self,
                 std_percent: float,
                 err_dep: EErrDep = EErrDep.INDEPENDENT,
                 seed: int | None = None) -> None:
        """
        Parameters
        ----------
        std_percent : float
            Standard deviation of the normal distribution to sample as a
            percentage.
        err_dep : EErrDependence, optional
            Error calculation dependence, by default EErrDependence.INDEPENDENT.
        seed : int | None, optional
            Optional seed for the random generator to allow for replicable
            behaviour, by default None.
        """
        self._std = np.abs(std_percent)/100
        self._rng = np.random.default_rng(seed)
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
        norm_rand = self._rng.normal(loc=0.0,
                                    scale=1.0,
                                    size=err_basis.shape)

        return (err_basis*self._std*norm_rand,sens_data)


class ErrRandGenerator(IErrCalculator):
    """Sensor random error calculator based on sampling a user specified random
    number generator implementing the `IGeneratorRandom` interface.

    Implements the `IErrCalculator` interface.
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
        rand_errs = self._generator.generate(shape=err_basis.shape)

        return (rand_errs,sens_data)


class ErrRandGenPercent(IErrCalculator):
    """Random error calculator based on sampling a user specified random
    number generator implementing the `IGeneratorRandom` interface. This class
    assumes the random generator is for a percentage error based on the input
    error basis and therefore it supports error dependence.

    The percentage error is calculated based on the ground truth if the error
    dependence is `INDEPENDENT` or based on the accumulated sensor measurement
    if the dependence is `DEPENDENT`.

    Implements the `IErrCalculator` interface.
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
        rand_errs = err_basis \
            * self._generator.generate(shape=err_basis.shape)/100.0

        return (rand_errs,sens_data)
