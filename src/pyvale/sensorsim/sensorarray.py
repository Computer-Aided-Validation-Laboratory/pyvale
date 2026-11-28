# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Self

import numpy as np

from pyvale.sensorsim.errorintegrator import ErrIntegrator
from pyvale.sensorsim.field import IField
from pyvale.sensorsim.fieldsamplerinterface import IFieldSampler
from pyvale.sensorsim.fieldscalar import FieldScalar
from pyvale.sensorsim.fieldtensor import FieldTensor
from pyvale.sensorsim.fieldvector import FieldVector
from pyvale.sensorsim.sensordata import SensorData
from pyvale.sensorsim.sensordescriptor import DescriptorFactory, SensorDescriptor


# TODO: when cameras, line, area and volume sensors are implemented this will
# need to be refactored.
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
    types of user selected errors (implementations of the `IErrSimulator`
    interface). Further information can be found in the `ErrIntegrator` class
    and in implementations of the `IErrSimulator` interface.

    In `pyvale`, function and methods with `calc` in their name will cause
    probability distributions to be resampled and any additional calculations,
    such as interpolation, to be performed. Functions and methods with `get` in
    the name will directly return the previously calculated values without
    resampling probability distributions.

    Calling the class method `sim_measurements()` will create and return an
    array of simulated sensor measurements with the following shape=(num_sensors
    ,num_field_component,num_time_steps). When calling `sim_measurements()` all
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
    def get_sensor_data(self) -> SensorData:
        """Return an independent copy of the nominal configuration."""

    @abstractmethod
    def set_sensor_data(self, sensor_data: SensorData) -> None:
        """Replace the nominal configuration and invalidate derived outputs."""

    @abstractmethod
    def set_field_sampler(self, sampler: IFieldSampler) -> None:
        """Replace the field sampler and invalidate derived outputs."""

    @abstractmethod
    def copy_for_simulation(self, *, seed: int | None = None) -> Self:
        """Copy mutable simulation state, optionally reseeding the copy."""

    @abstractmethod
    def get_measurement_shape(self) -> tuple[int, int, int]:
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

    @abstractmethod
    def get_field(self) -> IField:
        """Abstract method. Gets the field object that this array of sensors is
        sampling to simulate measurements.

        Returns
        -------
        IField
            A field object interface.
        """

    @abstractmethod
    def get_truth(self) -> np.ndarray:
        """Abstract method. Gets the ground truth sensor values that were
        calculated previously. If the ground truth values have not been
        calculated then `calc_truth()` is called first.

        Returns
        -------
        np.ndarray
            Array of ground truth sensor values. shape=(num_sensors,
            num_field_components,num_time_steps).
        """

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

    @abstractmethod
    def sim_measurements(self) -> np.ndarray:
        """Abstract method. Calculates measurements as: measurement = truth +
        systematic errors + random errors. The truth is calculated once and is
        interpolated from the input simulation field. The errors are calculated
        based on the user specified error chain.

        NOTE: this is a 'sim' method and will sample all probability
        distributions in the error chain returning a new simulated experiment
        for this sensor array.

        Returns
        -------
        np.ndarray
            The calculated measurements for this sensor array with shape:
            (num_sensors,num_field_components,num_time_steps)
        """

    @abstractmethod
    def get_measurements(self) -> np.ndarray:
        """Abstract method. Returns the current set of simulated measurements if
        theses have been calculated. If these have not been calculated then
        'sim_measurements()' is called and a set of measurements in then
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


def _resolve_sensor_descriptor(
    field: IField, descriptor: SensorDescriptor | None
) -> SensorDescriptor:
    """Use an explicit descriptor or build a default for the field type."""
    if descriptor is not None:
        return descriptor

    if isinstance(field, FieldScalar):
        return DescriptorFactory.scalar()

    if isinstance(field, FieldVector):
        return DescriptorFactory.vector()

    if isinstance(field, FieldTensor):
        return DescriptorFactory.tensor(field.get_spatial_dims())

    return SensorDescriptor()


def resolve_sample_times(sensor_data: SensorData, field: IField) -> np.ndarray:
    """Use the configured acquisition times or the field sampler's defaults."""
    if sensor_data.sample_times is not None:
        return sensor_data.sample_times

    return field.get_time_steps()


def _get_measurement_shape(
    sensor_data: SensorData, field: IField
) -> tuple[int, int, int]:
    positions = sensor_data.positions
    if positions is None or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Sensor positions must have shape (num_sensors, 3).")

    sample_times = resolve_sample_times(sensor_data, field)
    if sample_times.ndim != 1:
        raise ValueError("Sensor sample times must be one-dimensional.")

    return len(positions), len(field.get_all_components()), len(sample_times)


def _prepare_array_configuration(
    *,
    sensor_data: SensorData,
    field: IField,
    previous_field: IField,
    error_integrator: ErrIntegrator | None,
) -> tuple[SensorData, ErrIntegrator | None]:
    """Validate a layout and prepare error buffers before committing the update."""
    measurement_shape = _get_measurement_shape(sensor_data, field)
    updated_data = deepcopy(sensor_data)

    if error_integrator is not None:
        if field is not previous_field:
            # Redirect references to the old field while preserving generator state.
            error_integrator = deepcopy(error_integrator, {id(previous_field): field})

        error_integrator.set_sensor_data(
            updated_data, measurement_shape=measurement_shape
        )

    return updated_data, error_integrator
