'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import numpy as np
from pyvale.core.field import IField
from pyvale.core.sensor import ISensor
from pyvale.core.errorintegrator import ErrIntegrator
from pyvale.core.sensordescriptor import SensorDescriptor
from pyvale.core.sensordata import SensorData
from pyvale.core.fieldsampler import sample_field_with_sensor_data


class SensorArrayPoint(ISensor):
    __slots__ = ("field","descriptor","sensor_data","_truth","_measurements",
                 "error_integrator")

    def __init__(self,
                 sensor_data: SensorData,
                 field: IField,
                 descriptor: SensorDescriptor | None = None,
                 ) -> None:
        self.sensor_data = sensor_data
        self.field = field
        self.error_integrator = None

        self.descriptor = SensorDescriptor()
        if descriptor is not None:
            self.descriptor = descriptor

        self._truth = None
        self._measurements = None


    #---------------------------------------------------------------------------
    # accessors
    def get_sample_times(self) -> np.ndarray:
        if self.sensor_data.sample_times is None:
            #shape=(n_time_steps,)
            return self.field.get_time_steps()

        #shape=(n_time_steps,)
        return self.sensor_data.sample_times

    def get_measurement_shape(self) -> tuple[int,int,int]:
        return (self.sensor_data.positions.shape[0],
                len(self.field.get_all_components()),
                self.get_sample_times().shape[0])

    def get_field(self) -> IField:
        return self.field

    #---------------------------------------------------------------------------
    # Truth calculation from simulation
    def calc_truth_values(self) -> np.ndarray:
        self._truth = sample_field_with_sensor_data(self.field,
                                                    self.sensor_data)
        #shape=(n_sensors,n_field_comps,n_time_steps)
        return self._truth

    def get_truth(self) -> np.ndarray:
        if self._truth is None:
            self._truth = self.calc_truth_values()
        #shape=(n_sensors,n_field_comps,n_time_steps)
        return self._truth

    #---------------------------------------------------------------------------
    # Errors
    def set_error_integrator(self, err_int: ErrIntegrator) -> None:
        self.error_integrator = err_int

    def get_sensor_data_perturbed(self) -> SensorData | None:
        if self.error_integrator is None:
            return None

        return self.error_integrator.get_sens_data_accumulated()

    def get_errors_systematic(self) -> np.ndarray | None:
        if self.error_integrator is None:
            return None

        #shape=(n_sensors,n_field_comps,n_time_steps)
        return self.error_integrator.get_errs_systematic()

    def get_errors_random(self) -> np.ndarray | None:
        if self.error_integrator is None:
            return None

        #shape=(n_sensors,n_field_comps,n_time_steps)
        return self.error_integrator.get_errs_random()

    def get_errors_total(self) -> np.ndarray | None:
        if self.error_integrator is None:
            return None

        #shape=(n_sensors,n_field_comps,n_time_steps)
        return self.error_integrator.get_errs_total()

    #---------------------------------------------------------------------------
    # Measurements
    def calc_measurements(self) -> np.ndarray:
        if self.error_integrator is None:
            self._measurements = self.get_truth()
        else:
            self._measurements = self.get_truth() + \
                self.error_integrator.calc_errors_from_chain(self.get_truth())

        #shape=(n_sensors,n_field_comps,n_time_steps)
        return self._measurements

    def get_measurements(self) -> np.ndarray:
        if self._measurements is None:
            self._measurements = self.calc_measurements()

        #shape=(n_sensors,n_field_comps,n_time_steps)
        return self._measurements