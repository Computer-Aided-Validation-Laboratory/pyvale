# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

from typing import Callable
import numpy as np
from pyvale.errorcalculator import (IErrCalculator,
                                         EErrType,
                                         EErrDependence)
from pyvale.generatorsrandom import IGeneratorRandom
from pyvale.sensordata import SensorData


class ErrSysOffset(IErrCalculator):
    """Systematic error calculator applying a constant offset to all simulated
    sensor measurements. Implements the `IErrCalculator` interface.
    """
    __slots__ = ("_offset","_err_dep")

    def __init__(self,
                 offset: float,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT) -> None:
        """Initialiser for the `ErrSysOffset` class.

        Parameters
        ----------
        offset : float
            Constant offset to apply to all simulated measurements from the
            sensor array.
        err_dep : EErrDependence, optional
            Error , by default EErrDependence.INDEPENDENT
        """
        self._offset = offset
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
        """Gets the error dependence state for this error calculator. An
        independent error is calculated based on the input truth values as the
        error basis. A dependent error is calculated based on the accumulated
        sensor reading from all preceeding errors in the chain.

        NOTE: for this error the calculation is independent regardless of this
        setting as the offset is constant.

        Returns
        -------
        EErrDependence
            Enumeration defining INDEPENDENT or DEPENDENT behaviour.
        """
        return self._err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        """Sets the error dependence state for this error calculator. An
        independent error is calculated based on the input truth values as the
        error basis. A dependent error is calculated based on the accumulated
        sensor reading from all preceeding errors in the chain.

        NOTE: for this error the calculation is independent regardless of this
        setting as the offset is constant.

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
        return (self._offset*np.ones(shape=err_basis.shape),sens_data)


class ErrSysOffsetPercent(IErrCalculator):
    """Systematic error calculator applying a constant offset as a percentage of
    the sensor reading to each individual simulated sensor measurement.
    Implements the `IErrCalculator` interface.
    """
    __slots__ = ("_offset_percent","_err_dep")

    def __init__(self,
                 offset_percent: float,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT) -> None:
        """Initialiser for the `ErrSysOffsetPercent` class.

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

    def get_error_dep(self) -> EErrDependence:
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

    def set_error_dep(self, dependence: EErrDependence) -> None:
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
        return (self._offset_percent/100 *
                err_basis *
                np.ones(shape=err_basis.shape),
                sens_data)


class ErrSysUniform(IErrCalculator):
    """Systematic error calculator for applying an offset to each sensor that is
    sampled from a uniform probability distribution specified by its upper and
    lower bounds. Implements the `IErrCalculator` interface.
    """
    __slots__ = ("_low","_high","_rng","_err_dep")

    def __init__(self,
                 low: float,
                 high: float,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT,
                 seed: int | None = None) -> None:
        """Initialiser for the `ErrSysUniform` class.

        Parameters
        ----------
        low : float
            Lower bound of the uniform probability distribution in the same
            units as the physical field the sensor array is sampling.
        high : float
            Upper bound of the uniform probability distribution in the same
            units as the physical field the sensor array is sampling.
        err_dep : EErrDependence, optional
            Error calculation dependence, by default EErrDependence.INDEPENDENT.
        seed : int | None, optional
            Optional seed for the random generator to allow for replicable
            behaviour, by default None.
        """
        self._low = low
        self._high = high
        self._rng = np.random.default_rng(seed)
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
        """Gets the error dependence state for this error calculator. An
        independent error is calculated based on the input truth values as the
        error basis. A dependent error is calculated based on the accumulated
        sensor reading from all preceeding errors in the chain.

        NOTE: for this error the calculation is independent regardless of this
        setting as the offset is constant.

        Returns
        -------
        EErrDependence
            Enumeration defining INDEPENDENT or DEPENDENT behaviour.
        """
        return self._err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        """Sets the error dependence state for this error calculator. An
        independent error is calculated based on the input truth values as the
        error basis. A dependent error is calculated based on the accumulated
        sensor reading from all preceeding errors in the chain.

        NOTE: for this error the calculation is independent regardless of this
        setting as the offset is constant.

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
        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1
        sys_errs = self._rng.uniform(low=self._low,
                                    high=self._high,
                                    size=err_shape)

        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))

        return (sys_errs,sens_data)


class ErrSysUniformPercent(IErrCalculator):
    """Systematic error calculator for applying a percentage offset to each
    sensor that is sampled from a uniform probability distribution specified by
    its upper and lower bounds.

    The percentage offset is calculated based on the ground truth if the error
    dependence is `INDEPENDENT` or based on the accumulated sensor measurement
    if the dependence is `DEPENDENT`.

    Implements the `IErrCalculator` interface.
    """
    __slots__ = ("_low","_high","_rng","_err_dep")

    def __init__(self,
                 low_percent: float,
                 high_percent: float,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT,
                 seed: int | None = None) -> None:
        """_summary_

        Parameters
        ----------
        low_percent : float
            Lower percentage bound for the uniform probability distribution.
        high_percent : float
            Upper percentage bound for the uniform probability distribution.
        err_dep : EErrDependence, optional
            Error calculation dependence, by default EErrDependence.INDEPENDENT
        seed : int | None, optional
            Optional seed for the random generator to allow for replicable
            behaviour, by default None.
        """
        self._low = low_percent/100
        self._high = high_percent/100
        self._rng = np.random.default_rng(seed)
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
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

    def set_error_dep(self, dependence: EErrDependence) -> None:
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
        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1
        sys_errs = self._rng.uniform(low=self._low,
                                    high=self._high,
                                    size=err_shape)

        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))

        return (err_basis*sys_errs,sens_data)


class ErrSysNormal(IErrCalculator):
    """Systematic error calculator for applying an offset to each individual
    sensor in the array based on sampling from a normal distribution specified
    by its standard deviation and mean. Note that the offset is constant for
    each sensor over time. Implements the `IErrCalculator` interface.
    """
    __slots__ = ("_std","_rng","_err_dep")

    def __init__(self,
                 std: float,
                 mean: float = 0.0,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT,
                 seed: int | None = None) -> None:
        """Initialiser for the `ErrSysNormal` class.

        Parameters
        ----------
        std : float
            Standard deviation of the normal distribution to sample.
        mean : float, optional
            Mean of the normal distribution to sample, by default 0.0.
        err_dep : EErrDependence, optional
            Error calculation dependence, by default EErrDependence.INDEPENDENT.
        seed : int | None, optional
            Optional seed for the random generator to allow for replicable
            behaviour, by default None.
        """
        self._std = std
        self._mean = mean
        self._rng = np.random.default_rng(seed)
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
        """Gets the error dependence state for this error calculator. An
        independent error is calculated based on the input truth values as the
        error basis. A dependent error is calculated based on the accumulated
        sensor reading from all preceeding errors in the chain.

        NOTE: for this error the calculation is independent regardless of this
        setting as the offset is constant.

        Returns
        -------
        EErrDependence
            Enumeration defining INDEPENDENT or DEPENDENT behaviour.
        """
        return self._err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        """Sets the error dependence state for this error calculator. An
        independent error is calculated based on the input truth values as the
        error basis. A dependent error is calculated based on the accumulated
        sensor reading from all preceeding errors in the chain.

        NOTE: for this error the calculation is independent regardless of this
        setting as the offset is constant.

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
        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1
        sys_errs = self._rng.normal(loc=self._mean,
                                    scale=self._std,
                                    size=err_shape)

        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))

        return (sys_errs,sens_data)


class ErrSysNormPercent(IErrCalculator):
    """Systematic error calculator for applying a percentage offset to each
    individual sensor in the array based on sampling from a normal distribution
    specified by its standard deviation and mean. Note that the offset is
    constant for each sensor over time.

    The percentage offset is calculated based on the ground truth if the error
    dependence is `INDEPENDENT` or based on the accumulated sensor measurement
    if the dependence is `DEPENDENT`.

    Implements the `IErrCalculator` interface.
    """
    __slots__ = ("_std","_rng","_err_dep")

    def __init__(self,
                 std_percent: float,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT,
                 seed: int | None = None) -> None:
        self._std = std_percent/100
        self._rng = np.random.default_rng(seed)
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
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

    def set_error_dep(self, dependence: EErrDependence) -> None:
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

        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1
        sys_errs = self._rng.normal(loc=0.0,
                                    scale=self._std,
                                    size=err_shape)
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
        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))

        return (err_basis*sys_errs,sens_data)


class ErrSysGenerator(IErrCalculator):
    """Systematic error calculator for applying a unique offset to each sensor
    by sample from a user specified probability distribution (an implementation
    of the `IGeneratorRandom` interface).

    Implements the `IErrCalculator` interface.
    """
    __slots__ = ("_generator","_err_dep")

    def __init__(self,
                 generator: IGeneratorRandom,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT) -> None:

        self._generator = generator
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
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

    def set_error_dep(self, dependence: EErrDependence) -> None:
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
        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1

        sys_errs = self._generator.generate(shape=err_shape)

        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))

        return (sys_errs,sens_data)


class ErrSysGenPercent(IErrCalculator):
    """Systematic error calculator for applying a unique percentage offset to
    each sensor by sample from a user specified probability distribution (an
    implementation of the `IGeneratorRandom` interface). This class assumes the
    random generator is for a percentage error based on the input error basis
    and therefore it supports error dependence.

    The percentage error is calculated based on the ground truth if the error
    dependence is `INDEPENDENT` or based on the accumulated sensor measurement
    if the dependence is `DEPENDENT`.

    Implements the `IErrCalculator` interface.
    """
    __slots__ = ("_generator","_err_dep")

    def __init__(self,
                 generator: IGeneratorRandom,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT) -> None:

        self._generator = generator
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
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

    def set_error_dep(self, dependence: EErrDependence) -> None:
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
        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1

        sys_errs = self._generator.generate(size=err_shape)

        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))
        sys_errs = err_basis * sys_errs

        return (sys_errs,sens_data)


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
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT) -> None:
        """_summary_

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

    def get_error_dep(self) -> EErrDependence:
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

    def set_error_dep(self, dependence: EErrDependence) -> None:
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




