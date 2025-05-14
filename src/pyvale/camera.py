# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
NOTE: This module is a feature under developement.
"""

import numpy as np
from pyvale.field import IField
from pyvale.sensorarray import ISensorArray
from pyvale.errorintegrator import ErrIntegrator
from pyvale.sensordescriptor import SensorDescriptor
from pyvale.fieldsampler import sample_field_with_sensor_data
from pyvale.cameradata2d import CameraData2D
from pyvale.cameratools import CameraTools


class CameraBasic2D(ISensorArray):
    __slots__ = ("_cam_data","_field","_error_integrator","_descriptor",
                 "_sensor_data","_truth","_measurements")

    def __init__(self,
                 cam_data: CameraData2D,
                 field: IField,
                 descriptor: SensorDescriptor | None = None,
                 ) -> None:

        self._cam_data = cam_data
        self._field = field
        self._error_integrator = None

        self._descriptor = SensorDescriptor()
        if descriptor is not None:
            self._descriptor = descriptor

        self._sensor_data = CameraTools.build_sensor_data_from_camera_2d(self._cam_data)

        self._truth = None
        self._measurements = None

    #---------------------------------------------------------------------------
    # Accessors
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
    def calc_truth_values(self) -> np.ndarray:
        self._truth = sample_field_with_sensor_data(self._field,
                                                    self._sensor_data)
        #shape=(n_pixels,n_field_comps,n_time_steps)
        return self._truth

    def get_truth(self) -> np.ndarray:
        if self._truth is None:
            self._truth = self.calc_truth_values()
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
    def calc_measurements(self) -> np.ndarray:
        if self._error_integrator is None:
            self._measurements = self.get_truth()
        else:
            self._measurements = self.get_truth() + \
                self._error_integrator.calc_errors_from_chain(self.get_truth())

        #shape=(n_pixels,n_field_comps,n_time_steps)
        return self._measurements

    def get_measurements(self) -> np.ndarray:
        if self._measurements is None:
            self._measurements = self.calc_measurements()

        #shape=(n_pixels,n_field_comps,n_time_steps)
        return self._measurements

    #---------------------------------------------------------------------------
    # Images
    def calc_measurement_images(self) -> np.ndarray:
        #shape=(n_pixels,n_field_comps,n_time_steps)
        self._measurements = self.calc_measurements()
        image_shape = self.get_image_measurements_shape()
        #shape=(n_pixels_y,n_pixels_x,n_field_comps,n_time_steps)
        return np.reshape(self._measurements,image_shape)

    def get_measurement_images(self) -> np.ndarray:
        self._measurements = self.get_measurements()
        image_shape = self.get_image_measurements_shape()
        #shape=(n_pixels_y,n_pixels_x,n_field_comps,n_time_steps)
        return np.reshape(self._measurements,image_shape)



