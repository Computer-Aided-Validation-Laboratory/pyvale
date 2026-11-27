# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
NOTE: This module is a feature under developement.
"""

import copy

import numpy as np
from pyvale.sensorsim.field import IField
from pyvale.sensorsim.sensorarray import ISensorArray
from pyvale.sensorsim.sensordata import SensorData
from pyvale.sensorsim.errorintegrator import ErrIntegrator
from pyvale.sensorsim.sensordescriptor import SensorDescriptor
from pyvale.sensorsim.fieldsampler import sample_field_with_sensor_data
from pyvale.render.camera import Camera
from pyvale.sensorsim.cameratools import CameraTools



class CameraBasic2D(ISensorArray):
    __slots__ = ("_cam_data","_field","_error_integrator","_descriptor",
                 "_sensor_data","_truth","_measurements")

    def __init__(self,
                 cam_data: Camera,
                 field: IField,
                 descriptor: SensorDescriptor | None = None,
                 ) -> None:

        self._cam_data = cam_data
        self._field = field
        self._error_integrator = None

        self._descriptor = SensorDescriptor()
        if descriptor is not None:
            self._descriptor = descriptor

        self._sensor_data = (
            CameraTools.build_sensor_data_from_camera_2d(self._cam_data)
        )

        self._truth = None
        self._measurements = None

    #---------------------------------------------------------------------------
    # Accessors
    def get_sensor_data(self) -> SensorData:
        """Return a copy of the nominal sensor configuration."""
        return copy.deepcopy(self._sensor_data)

    def set_sensor_data(self, sensor_data: SensorData) -> None:
        """Apply a nominal configuration and clear outputs from the old layout."""
        positions = sensor_data.positions
        if positions is None or positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Sensor positions must have shape (num_sensors, 3).")

        if positions.shape != self._sensor_data.positions.shape:
            raise ValueError("The new layout must preserve the camera's pixel count.")

        updated_data = copy.deepcopy(sensor_data)
        self._clear_cached_outputs()
        self._sensor_data = updated_data
        if self._error_integrator is not None:
            self._error_integrator.set_sensor_data(
                updated_data, measurement_shape=self.get_measurement_shape()
            )

    def copy_for_simulation(self, *, seed: int | None = None) -> "CameraBasic2D":
        """Return an independent working array, preserving field and error links."""
        working_array = copy.deepcopy(self)
        working_array.set_sensor_data(self._sensor_data)
        if seed is not None and working_array._error_integrator is not None:
            working_array._error_integrator.reseed_error_chain(seed)
        return working_array

    def _clear_cached_outputs(self) -> None:
        self._truth = None
        self._measurements = None

    def get_sensor_positions(self) -> np.ndarray | None:
        """Return the configured sensor positions before perturbation."""
        return self._sensor_data.positions

    def get_sample_times(self) -> np.ndarray:
        if self._sensor_data.sample_times is None:
            #shape=(n_time_steps,)
            return self._field.get_time_steps()

        #shape=(n_time_steps,)
        return self._sensor_data.sample_times

    def get_measurement_shape(self) -> tuple[int,int,int]:
        return (self._sensor_data.positions.shape[0],
                len(self._field.get_all_components()),
                self.get_sample_times().shape[0])

    def get_image_measurements_shape(self) -> tuple[int,int,int,int]:
        return (self._cam_data.num_pixels[1],
                self._cam_data.num_pixels[0],
                len(self._field.get_all_components()),
                self.get_sample_times().shape[0])

    def get_field(self) -> IField:
        return self._field

    def get_descriptor(self) -> SensorDescriptor:
        return self._descriptor

    #---------------------------------------------------------------------------
    # Truth calculation from simulation
    def calc_truth(self) -> np.ndarray:
        self._truth = sample_field_with_sensor_data(self._field,
                                                    self._sensor_data)
        #shape=(n_pixels,n_field_comps,n_time_steps)
        return self._truth

    def get_truth(self) -> np.ndarray:
        if self._truth is None:
            self._truth = self.calc_truth()
        #shape=(n_pixels,n_field_comps,n_time_steps)
        return self._truth

    #---------------------------------------------------------------------------
    # Errors
    def set_error_integrator(self, err_int: ErrIntegrator) -> None:
        self._error_integrator = err_int

    def get_errors_systematic(self) -> np.ndarray | None:
        if self._error_integrator is None:
            return None

        #shape=(n_pixels,n_field_comps,n_time_steps)
        return self._error_integrator.get_errs_systematic()

    def get_errors_random(self) -> np.ndarray | None:
        if self._error_integrator is None:
            return None

        #shape=(n_pixels,n_field_comps,n_time_steps)
        return self._error_integrator.get_errs_random()

    def get_errors_total(self) -> np.ndarray | None:
        if self._error_integrator is None:
            return None

        #shape=(n_pixels,n_field_comps,n_time_steps)
        return self._error_integrator.get_errs_total()

    #---------------------------------------------------------------------------
    # Measurements
    def sim_measurements(self) -> np.ndarray:
        if self._error_integrator is None:
            self._measurements = self.get_truth()
        else:
            self._measurements = self.get_truth() + \
                self._error_integrator.calc_errors_from_chain(self.get_truth())

        #shape=(n_pixels,n_field_comps,n_time_steps)
        return self._measurements

    def get_measurements(self) -> np.ndarray:
        if self._measurements is None:
            self._measurements = self.sim_measurements()

        #shape=(n_pixels,n_field_comps,n_time_steps)
        return self._measurements

    #---------------------------------------------------------------------------
    # Images
    def calc_measurement_images(self) -> np.ndarray:
        #shape=(n_pixels,n_field_comps,n_time_steps)
        self._measurements = self.sim_measurements()
        image_shape = self.get_image_measurements_shape()
        #shape=(n_pixels_y,n_pixels_x,n_field_comps,n_time_steps)
        return np.reshape(self._measurements,image_shape)

    def get_measurement_images(self) -> np.ndarray:
        self._measurements = self.get_measurements()
        image_shape = self.get_image_measurements_shape()
        #shape=(n_pixels_y,n_pixels_x,n_field_comps,n_time_steps)
        return np.reshape(self._measurements,image_shape)


