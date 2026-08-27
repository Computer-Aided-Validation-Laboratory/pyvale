# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
NOTE: This module is a feature under developement.
"""

import numpy as np
from pyvale.sensorsim.field import IField
from pyvale.sensorsim.sensorarray import ISensorArray
from pyvale.sensorsim.errorintegrator import ErrIntegrator
from pyvale.sensorsim.sensordescriptor import SensorDescriptor
from pyvale.sensorsim.sensordata import SensorData
from pyvale.sensorsim.fieldsampler import sample_field_with_sensor_data
from pyvale.render.camera import Camera2D
from pyvale.render.cameratools import pixel_grid_leng



class CameraBasic2D(ISensorArray):
    __slots__ = ("_cam_data","_field","_error_integrator","_descriptor",
                 "_sensor_data","_truth","_measurements")

    def __init__(self,
                 cam_data: Camera2D,
                 field: IField,
                 descriptor: SensorDescriptor | None = None,
                 ) -> None:

        self._cam_data = cam_data
        self._field = field
        self._error_integrator = None

        self._descriptor = SensorDescriptor()
        if descriptor is not None:
            self._descriptor = descriptor

        pixel_grid = pixel_grid_leng(
            self._cam_data.field_of_view,
            self._cam_data.pixels_size,
        )
        positions = np.zeros((pixel_grid[0].size, 3))
        positions[:, :2] = np.column_stack(
            (pixel_grid[0].ravel(), pixel_grid[1].ravel())
        ) - self._cam_data.world_to_cam
        angles = (
            None
            if self._cam_data.angle is None
            else (self._cam_data.angle,)
        )
        self._sensor_data = SensorData(
            positions=positions,
            sample_times=self._cam_data.sample_times,
            angles=angles,
        )

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
        return (self._cam_data.pixels_num[1],
                self._cam_data.pixels_num[0],
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

