# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import copy
from dataclasses import dataclass
import numpy as np
from pyvale.errorcalculator import (IErrCalculator,
                                    EErrType,
                                    EErrDep)
from pyvale.sensordata import SensorData


@dataclass(slots=True)
class ErrIntOpts:
    """Error integration options dataclass. Allows the user to control how
    errors are calculated and stored in memory for later use.
    """

    force_dependence: EErrDep | None = None
    """Forces all errors to be calculated with the specified dependence. If set
    to None then all errors will use their default/preset dependence.

    Note that some errors are inherently independent so will not change. For
    example: `ErrRandNormal` is purely independent whereas `ErrRandNormPercent`
    can have the percentage error calculated based on the ground truth
    (independent) or based on the accumulated sensor measurement (dependent).
    """

    store_all_errs: bool = False
    """Stores all errors for individual error in the chain if True. Also stores
    a list of SensorData objects showing perturbations to the sensor array
    parameters caused by individual errors. Consumes significantly more memory
    but is useful for finding which errors contribute most to the total
    measurement error. For large sensor arrays (>100 sensors)
    """


class ErrIntegrator:
    """Class for managing sensor error integration. Takes a list of objects that
    implement the `IErrCalculator` interface (i.e. the error chain) and loops
    through them calculating each errors contribution to the total measurement
    error and sums this over all errors in the chain. In addition to the total
    error a sum of the random and systematic errors (see `EErrType`) is
    calculated and stored.

    This class also accumulates perturbations to the sensor array parameters due
    to errors (i.e. sensor positioning error or temporal drift). The accumulated
    sensor array parameters are stored as a `SensorData` object.

    The errors are calculated in the order specified in the list. For dependent
    errors (`EErrDependence.DEPENDENT`) the position of the error within the
    error chain determines the accumulated sensor measurement that will be used
    to calculate the error.

    The user can control how the errors are calculated using the `ErrIntOpts`
    dataclass.
    """
    __slots__ = ("_err_chain","_meas_shape","_errs_by_chain",
                 "_errs_systematic","_errs_random","_errs_total",
                 "_sens_data_by_chain","_err_int_opts","_sens_data_accumulated",
                 "_sens_data_initial")

    def __init__(self,
                 err_chain: list[IErrCalculator],
                 sensor_data_initial: SensorData,
                 meas_shape: tuple[int,int,int],
                 err_int_opts: ErrIntOpts | None = None) -> None:
        """
        Parameters
        ----------
        err_chain : list[IErrCalculator]
            List of error objects implementing the IErrCalculator interface.
        sensor_data_initial : SensorData
            Object holding the initial sensor array parameters before they are
            modified by the error chain.
        meas_shape : tuple[int,int,int]
            Shape of the sensor measurement array. shape=(num_sensors,
            num_field_components,num_time_steps)
        err_int_opts : ErrIntOpts | None, optional
            Options for controlling how errors are calculated/summed and how
            they are store in memory, by default None. If None then the default
            options dataclass is used.
        """

        if err_int_opts is None:
            self._err_int_opts = ErrIntOpts()
        else:
            self._err_int_opts = err_int_opts

        self.set_error_chain(err_chain)
        self._meas_shape = meas_shape

        self._sens_data_initial = copy.deepcopy(sensor_data_initial)
        self._sens_data_accumulated = copy.deepcopy(sensor_data_initial)

        if self._err_int_opts.store_all_errs:
            self._sens_data_by_chain = []
            self._errs_by_chain = np.zeros((len(self._err_chain),)+ \
                                               self._meas_shape)
        else:
            self._sens_data_by_chain = None
            self._errs_by_chain = None

        self._errs_systematic = np.zeros(meas_shape)
        self._errs_random = np.zeros(meas_shape)
        self._errs_total = np.zeros(meas_shape)


    def set_error_chain(self, err_chain: list[IErrCalculator]) -> None:
        """Sets the error chain that will be looped over to calculate the sensor
        measurement errors. If the error integration options are forcing error
        dependence then all errors in the chain will have their dependence set
        to the specified value.

        Parameters
        ----------
        err_chain : list[IErrCalculator]
            List of error calculators implementing the IErrCalculator interface.
        """
        self._err_chain = err_chain

        if self._err_int_opts.force_dependence is not None:
            for ee in self._err_chain:
                ee.set_error_dep(self._err_int_opts.force_dependence)


    def calc_errors_from_chain(self, truth: np.ndarray) -> np.ndarray:
        """Calculates all errors by looping over the error chain. The total
         measurement error is summed as each error is calculated in order. Note
         that this causes all errors based on probability distributions to be
         resampled and any required interpolations to be performed (e.g. from
         randomly perturbing the sensor positions). Accumulated errors are also
         stored for random and systematic errors separately (see `EErrType`).

         If the `store_all_errs = True` in the `ErrIntOpts` dataclass then each
         individual error is stored in a numpy array (see `get_errs_by_chain()`)
         along with the accumulated errors in another numpy array.

        Parameters
        ----------
        truth : np.ndarray
            Array of ground truth sensor measurements interpolated from the
            simulated physical field. shape=(num_sensors,num_field_components,
            num_time_steps).

        Returns
        -------
        np.ndarray
            Array of total errors summed over all errors in the chain. shape=(
            num_sensors,num_field_components,num_time_steps).
        """
        if self._err_int_opts.store_all_errs:
            return self._calc_errors_store_by_chain(truth)

        return self._calc_errors_mem_eff(truth)


    def _calc_errors_store_by_chain(self, truth: np.ndarray) -> np.ndarray:
        """Helper function for calculating all errors in the chain and summing
        them. Returns the total error and stores sums of the random and
        systematic errors in member variables. This function also stores each
        individual error calculation in a separate numpy array for analysis.

        Parameters
        ----------
        truth : np.ndarray
            Array of ground truth sensor measurements interpolated from the
            simulated physical field. shape=(num_sensors,num_field_components,
            num_time_steps).

        Returns
        -------
        np.ndarray
            Array of total errors summed over all errors in the chain. shape=(
            num_sensors,num_field_components,num_time_steps).
        """
        accumulated_error = np.zeros_like(truth)
        self._errs_by_chain = np.zeros((len(self._err_chain),) + \
                                           self._meas_shape)

        for ii,ee in enumerate(self._err_chain):

            if ee.get_error_dep() == EErrDep.DEPENDENT:
                (error_array,sens_data) = ee.calc_errs(truth+accumulated_error,
                                                       self._sens_data_accumulated)

            else:
                (error_array,sens_data) = ee.calc_errs(truth,
                                                       self._sens_data_initial)

            self._sens_data_accumulated = sens_data
            self._sens_data_by_chain.append(sens_data)

            if ee.get_error_type() == EErrType.SYSTEMATIC:
                self._errs_systematic = self._errs_systematic + error_array
            else:
                self._errs_random = self._errs_random + error_array

            accumulated_error = accumulated_error + error_array
            self._errs_by_chain[ii,:,:,:] = error_array

        self._errs_total = accumulated_error
        return self._errs_total


    def _calc_errors_mem_eff(self, truth: np.ndarray) -> np.ndarray:
        """Helper function for calculating all errors in the chain and summing
        them. Returns the total error and stores sums of the random and
        systematic errors in member variables. The individual error
        contributions are not stored in this case for memory efficiency, only
        the summed total, random and systematic error arrays are stored.

        Parameters
        ----------
        truth : np.ndarray
            Array of ground truth sensor measurements interpolated from the
            simulated physical field. shape=(num_sensors,num_field_components,
            num_time_steps).

        Returns
        -------
        np.ndarray
            Array of total errors summed over all errors in the chain. shape=(
            num_sensors,num_field_components,num_time_steps).
        """
        accumulated_error = np.zeros_like(truth)

        for ee in self._err_chain:

            if ee.get_error_dep() == EErrDep.DEPENDENT:
                (error_array,sens_data) = ee.calc_errs(
                    truth+accumulated_error,
                    self._sens_data_accumulated
                )
            else:
                (error_array,sens_data) = ee.calc_errs(truth,
                                                       self._sens_data_initial)

            self._sens_data_accumulated = sens_data

            if ee.get_error_type() == EErrType.SYSTEMATIC:
                self._errs_systematic = self._errs_systematic + error_array
            else:
                self._errs_random = self._errs_random + error_array

            accumulated_error = accumulated_error + error_array

        self._errs_total = accumulated_error
        return self._errs_total


    def get_errs_by_chain(self) -> np.ndarray | None:
        """Gets the array of errors for each error in chain. If `store_all_errs`
        is False in `ErrIntOpts` then this will return None.

        Returns
        -------
        np.ndarray | None
            Array of all errors in the chain. shape=(num_errs_in_chain,
            num_sensors,num_field_components,num_time_steps). Returns None if
            `ErrIntOpts.store_all_errs=False`.
        """
        return self._errs_by_chain

    def get_sens_data_by_chain(self) -> list[SensorData] | None:
        """Gets the list of sensor data objects storing how each error in the
        chain has perturbed the underlying sensor parameters. If
        `store_all_errs` is False in `ErrIntOpts` then this will return None.
        If no sensor array parameters are modified by the error chain then all
        SensorData objects in the list will be identical to the SensorData
        object used to create the sensor array.

        Returns
        -------
        list[SensorData] | None
            List of perturbed sensors array parameters for each error in the
            chain. Returns None if `ErrIntOpts.store_all_errs=False`.
        """
        return self._sens_data_by_chain

    def get_sens_data_accumulated(self) -> SensorData:
        """Gets the final accumulated sensor array parameters based on all
        errors in the chain as a SensorData object. If no errors modify the
        sensor array parameters then the SensorData object returns will be
        identical to the SensorData object used to create the sensor array.

        Returns
        -------
        SensorData
            The final sensor array parameters based on accumulating all
            perturbations from all errors in the error chain.
        """
        return self._sens_data_accumulated

    def get_errs_systematic(self) -> np.ndarray:
        """Gets the array of summed systematic errors over the error chain. If
        the errors have not been calculated then an array of zeros is returned.

        Returns
        -------
        np.ndarray
            Array of total systematic errors. shape=(num_sensors,
            num_field_components,num_time_steps)
        """
        return self._errs_systematic

    def get_errs_random(self) -> np.ndarray:
        """Gets the array of summed random errors over the error chain. If the
        errors have not been calculated then an array of zeros is returned.

        Returns
        -------
        np.ndarray
            Array of total random errors. shape=(num_sensors,
            num_field_components,num_time_steps)
        """
        return self._errs_random

    def get_errs_total(self) -> np.ndarray:
        """Gets the array of total errors. If the errors have not been
        calculated then an array of zeros is returned. Note that this function
        just returns the most recently calculated errors and will not resample
        from probability distributions.

        Returns
        -------
        np.ndarray
            Array of total errors. shape=(num_sensors,num_field_components,
            num_time_steps)
        """
        return self._errs_total


