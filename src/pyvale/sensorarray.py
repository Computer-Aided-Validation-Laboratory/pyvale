# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from abc import ABC, abstractmethod
import numpy as np
from pyvale.field import IField


class ISensorArray(ABC):
    """Interface (abstract base class) for an array of sensors of the same
    type sampling a given physical field.

    This class implements the `pyvale` sensor measurement simulation model. Here
    a measurement is taken as: measurement = truth + random errors + systematic
    errors. The truth value for each sensor is interpolated from the physical
    field (an implementation of the `IField` interface, nominally a
    `FieldScalar`, `FieldVector` or `FieldTensor` object).

    The random and systematic errors are calculated by a user specified error
    integrator (`ErrIntegrator` class). This class contains a chain of different
    types of user selected errors (implementations of the `IErrCalculator`
    interface). Further information can be found in the `ErrIntegrator` class
    and in implementations of the `IErrCalculator` interface.

    In `pyvale`, function and methods with `calc` in their name will cause
    probability distributions to be resampled and any additional calculations,
    such as interpolation, to be performed. Functions and methods with `get` in
    the name will directly return the previously calculated values without
    resampling probability distributions.

    Calling the class method `calc_measurements()` will create and return an
    array of simulated sensor measurements with the following shape=(num_sensors
    ,num_field_component,num_time_steps). When calling `calc_measurements()` all
    sensor errors that are based on probability distributions are resampled and
    any required interpolations are performed (e.g. a random perturbation of the
    sensor positions requiring interpolation at the perturbed sensor location).

    Calling the class method `get_measurements()` just returns the previously
    calculated set of sensor measurements without resampling of probability.
    Distributions.

    Without an error integrator this class can be used for interpolating
    simulated physical fields quickly using finite element shape functions.
    """

    @abstractmethod
    def get_measurement_shape(self) -> tuple[int,int,int]:
        """Abstract method. Gets the shape of the measurement array:
        shape=(num_sensors,num_field_components,num_time_steps).

        The number of sensors is specified by the user with a SensorData object.
        The number of field components is dependent on the field being sampled
        (i.e. 1 for a scalar field and 3 for a vector field in 3D). The number
        of time steps is specified by the user in the SensorData object or
        defaults to the time steps taken from the simulation.

        Returns
        -------
        tuple[int,int,int]
            Shape of the measurement array as (num_sensors,
            num_field_components,num_time_steps)
        """
        pass

    @abstractmethod
    def get_field(self) -> IField:
        """Abstract method. Gets the field object that this array of sensors is
        sampling to simulate measurements.

        Returns
        -------
        IField
            A field object interface.
        """
        pass

    @abstractmethod
    def get_truth(self) -> np.ndarray:
        """Abstract method. Gets the ground truth sensor values that were
        calculated previously. If the ground truth values have not been
        calculated then `calc_truth_values()` is called first.

        Returns
        -------
        np.ndarray
            Array of ground truth sensor values. shape=(num_sensors,
            num_field_components,num_time_steps).
        """
        pass

    @abstractmethod
    def get_errors_systematic(self) -> np.ndarray | None:
        """Abstract method. Gets the systematic error array from the previously
        calculated sensor measurements. Returns None is no error integrator has
        been specified.

        Returns
        -------
        np.ndarray | None
            Array of systematic errors for this sensor array. shape=(num_sensors
            ,num_field_components,num_time_steps). Returns None if no error
            integrator has been set.
        """
        pass

    @abstractmethod
    def get_errors_random(self) -> np.ndarray | None:
        """Abstract method. Gets the random error array from the previously
        calculated sensor measurements. Returns None is no error integrator has
        been specified.

        Returns
        -------
        np.ndarray | None
            Array of random errors for this sensor array. shape=(num_sensors
            ,num_field_components,num_time_steps). Returns None if no error
            integrator has been set.
        """
        pass

    @abstractmethod
    def get_errors_total(self) -> np.ndarray | None:
        """Abstract method. Gets the total error array from the previously
        calculated sensor measurements. Returns None is no error integrator has
        been specified.

        Returns
        -------
        np.ndarray | None
            Array of total errors for this sensor array. shape=(num_sensors
            ,num_field_components,num_time_steps). Returns None if no error
            integrator has been set.
        """
        pass

    @abstractmethod
    def calc_measurements(self) -> np.ndarray:
        """Abstract method. Calculates measurements as: measurement = truth +
        systematic errors + random errors. The truth is calculated once and is
        interpolated from the input simulation field. The errors are calculated
        based on the user specified error chain.

        NOTE: this is a 'calc' method and will sample all probability
        distributions in the error chain returning a new simulated experiment
        for this sensor array.

        Returns
        -------
        np.ndarray
            The calculated measurements for this sensor array with shape:
            (num_sensors,num_field_components,num_time_steps)
        """
        pass

    @abstractmethod
    def get_measurements(self) -> np.ndarray:
        """Abstract method. Returns the current set of simulated measurements if
        theses have been calculated. If these have not been calculated then
        'calc_measurements()' is called and a set of measurements in then
        returned.

        NOTE: this is a 'get' method and does not sample from probability
        distributions in the error chain and directly returns the current set of
        measurements if they exist.

        Returns
        -------
        np.ndarray
            The calculated measurements for this sensor array with shape:
            (num_sensors,num_field_components,num_time_steps)
        """
        pass



