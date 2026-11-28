# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
NOTE: This module is a feature under developement.
"""

import copy
from typing import Self

import numpy as np

from pyvale.render.camera import Camera
from pyvale.sensorsim.cameratools import CameraTools
from pyvale.sensorsim.errorintegrator import ErrIntegrator
from pyvale.sensorsim.field import IField
from pyvale.sensorsim.fieldsampler import sample_field_with_sensor_data
from pyvale.sensorsim.fieldsamplerinterface import IFieldSampler
from pyvale.sensorsim.sensorarray import (
    ISensorArray,
    _get_measurement_shape,
    _prepare_array_configuration,
    _resolve_sensor_descriptor,
)
from pyvale.sensorsim.sensordata import SensorData
from pyvale.sensorsim.sensordescriptor import SensorDescriptor


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

        self._descriptor = _resolve_sensor_descriptor(field, descriptor)

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
        current_positions = self._sensor_data.positions
        new_positions = sensor_data.positions
        if (
            current_positions is None
            or new_positions is None
            or new_positions.shape != current_positions.shape
        ):
            raise ValueError("The new layout must preserve the camera's pixel count.")

        self._apply_configuration(sensor_data=sensor_data, field=self._field)

    def set_field_sampler(self, sampler: IFieldSampler) -> None:
        """Install a replacement field and refresh measurement and error state."""
        field = self._field.with_sampler(sampler)
        self._apply_configuration(sensor_data=self._sensor_data, field=field)

    def _apply_configuration(self, *, sensor_data: SensorData, field: IField) -> None:
        updated_data, error_integrator = _prepare_array_configuration(
            sensor_data=sensor_data,
            field=field,
            previous_field=self._field,
            error_integrator=self._error_integrator,
        )

        self._sensor_data = updated_data
        self._field = field
        self._error_integrator = error_integrator
        self._truth = None
        self._measurements = None

    def copy_for_simulation(self, *, seed: int | None = None) -> Self:
        """Return an independent working array, preserving field and error links."""
        working_array = copy.deepcopy(self)
        working_array.set_sensor_data(self._sensor_data)

        if seed is not None and working_array._error_integrator is not None:
            working_array._error_integrator.reseed_error_chain(seed)

        return working_array

    def get_measurement_shape(self) -> tuple[int, int, int]:
        """Return the (sensor, component, time) measurement shape."""
        return _get_measurement_shape(self._sensor_data, self._field)

    def get_image_measurements_shape(self) -> tuple[int, int, int, int]:
        _, num_components, num_times = self.get_measurement_shape()
        return (
            self._cam_data.num_pixels[1],
            self._cam_data.num_pixels[0],
            num_components,
            num_times,
        )

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


