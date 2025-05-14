# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np
from pyvale.field import IField
from pyvale.sensorarray import ISensorArray
from pyvale.errorintegrator import ErrIntegrator
from pyvale.sensordescriptor import SensorDescriptor
from pyvale.sensordata import SensorData
from pyvale.fieldsampler import sample_field_with_sensor_data


class SensorArrayPoint(ISensorArray):
    """A class for creating arrays of point sensors applied to a simulated
    physical field. Examples include: thermocouples used to measure temperature
    (a scalar field) or strain gauges used to measure strain (a tensor field).
    Implements the ISensorArray interface.

    This class uses the `pyvale` sensor measurement simulation model. Here a
    measurement is taken as: measurement = truth + random errors + systematic
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

    __slots__ = ("_field","_descriptor","_sensor_data","_truth","_measurements",
                 "_error_integrator")

    def __init__(self,
                 sensor_data: SensorData,
                 field: IField,
                 descriptor: SensorDescriptor | None = None,
                 ) -> None:
        """
        Parameters
        ----------
        sensor_data : SensorData
            Specifies sensor array parameters including: positions, sample times
            , angles, and area averaging. See the `SensorData` dataclass for
            details.
        field : IField
            The simulated physical field that the sensors will samples from.
            This is normally a `FieldScalar`, `FieldVector` or `FieldTensor`.
        descriptor : SensorDescriptor | None, optional
            Contains descriptive information about the sensor array for display
            and visualisations, by default None.
        """
        self._sensor_data = sensor_data
        self._field = field
        self._error_integrator = None

        self._descriptor = SensorDescriptor()
        if descriptor is not None:
            self._descriptor = descriptor

        self._truth = None
        self._measurements = None

    def get_sample_times(self) -> np.ndarray:
        """Gets the times at which the sensors sample the given physical field.
        This is specified by the user in the SensorData object or defaults to
        the time steps in the underlying simulation if unspecified.

        Returns
        -------
        np.ndarray
            Sample times with shape: (num_time_steps,)
        """
        if self._sensor_data.sample_times is None:
            return self._field.get_time_steps()

        return self._sensor_data.sample_times

    def get_measurement_shape(self) -> tuple[int,int,int]:
        """Gets the shape of the sensor measurement array. shape=(num_sensors,
        num_field_components,num_time_steps)

        Returns
        -------
        tuple[int,int,int]
            Shape of the measurement array. shape=(num_sensors,
            num_field_components,num_time_steps)
        """

        return (self._sensor_data.positions.shape[0],
                len(self._field.get_all_components()),
                self.get_sample_times().shape[0])

    def get_field(self) -> IField:
        """Gets a reference to the physical field that this sensor array
        is applied to.

        Returns
        -------
        IField
            Reference to an `IField` interface.
        """
        return self._field


    def calc_truth_values(self) -> np.ndarray:
        """Calculates the ground truth sensor values by interpolating the
        simulated physical field using the sensor array parameters in the
        `SensorData` object.

        Returns
        -------
        np.ndarray
            Array of ground truth sensor values. shape=(num_sensors,
            num_field_components,num_time_steps).
        """
        self._truth = sample_field_with_sensor_data(self._field,
                                                    self._sensor_data)

        return self._truth

    def get_truth(self) -> np.ndarray:
        """Gets the ground truth sensor values that were calculated previously.
        If the ground truth values have not been calculated then
        `calc_truth_values()` is called first.

        Returns
        -------
        np.ndarray
            Array of ground truth sensor values. shape=(num_sensors,
            num_field_components,num_time_steps).
        """
        if self._truth is None:
            self._truth = self.calc_truth_values()

        return self._truth

    def set_error_integrator(self, err_int: ErrIntegrator) -> None:
        """Sets the error intergrator that will be used to calculate the sensor
        array measurement errors when `calc_measurements()` is called. See the
        `ErrIntegrator` class for further detail.

        Parameters
        ----------
        err_int : ErrIntegrator
            Error integration object with a chain of user defined sensor errors.
        """
        self._error_integrator = err_int

    def get_sensor_data_perturbed(self) -> SensorData | None:
        """Gets the final sensor array parameters after all errors in the error
        integrator have been applied. If no error integrator is specified then
        None is returned.

        Returns
        -------
        SensorData | None
            The accumulated sensor array parameters as a SensorData object.
            Returns None if no error integrator has been specified.
        """
        if self._error_integrator is None:
            return None

        return self._error_integrator.get_sens_data_accumulated()

    def get_errors_systematic(self) -> np.ndarray | None:
        """Gets the systematic error array from the previously calculated sensor
        measurements. Returns None is no error integrator has been specified.

        Returns
        -------
        np.ndarray | None
            Array of systematic errors for this sensor array. shape=(num_sensors
            ,num_field_components,num_time_steps). Returns None if no error
            integrator has been set.
        """
        if self._error_integrator is None:
            return None

        return self._error_integrator.get_errs_systematic()

    def get_errors_random(self) -> np.ndarray | None:
        """Gets the random error array from the previously calculated sensor
        measurements. Returns None is no error integrator has been specified.

        Returns
        -------
        np.ndarray | None
            Array of random errors for this sensor array. shape=(num_sensors
            ,num_field_components,num_time_steps). Returns None if no error
            integrator has been set.
        """
        if self._error_integrator is None:
            return None

        return self._error_integrator.get_errs_random()

    def get_errors_total(self) -> np.ndarray | None:
        """Gets the total error array from the previously calculated sensor
        measurements. Returns None is no error integrator has been specified.

        Returns
        -------
        np.ndarray | None
            Array of total errors for this sensor array. shape=(num_sensors
            ,num_field_components,num_time_steps). Returns None if no error
            integrator has been set.
        """
        if self._error_integrator is None:
            return None

        return self._error_integrator.get_errs_total()

    def calc_measurements(self) -> np.ndarray:
        """Calculates a set of sensor measurements using the specified sensor
        array parameters and the error intergator if specified. Calculates
        measurements as: measurement = truth + systematic errors + random errors
        . The truth is calculated once and is interpolated from the input
        simulation field. The errors are calculated based on the user specified
        error chain in the error integrator object. If no error integrator is
        specified then only the truth is returned.            _description_ew simulated experiment
        for this sensor array.

        Returns
        -------
        np.ndarray
            Array of sensor measurements including any simulated random and
            systematic errors if an error integrator is specified. shape=(
            num_sensors,num_field_components,num_time_steps).
        """
        if self._error_integrator is None:
            self._measurements = self.get_truth()
        else:
            self._measurements = self.get_truth() + \
                self._error_integrator.calc_errors_from_chain(self.get_truth())

        return self._measurements

    def get_measurements(self) -> np.ndarray:
        """Returns the current set of simulated measurements if theses have been
        calculated. If these have not been calculated then 'calc_measurements()'
        is called and a set of measurements in then returned.

        Returns
        -------
        np.ndarray
            Array of sensor measurements including any simulated random and
            systematic errors if an error integrator is specified. shape=(
            num_sensors,num_field_components,num_time_steps).
        """
        if self._measurements is None:
            self._measurements = self.calc_measurements()

        return self._measurements
