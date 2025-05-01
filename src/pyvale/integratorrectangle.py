# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import numpy as np
from pyvale.field import IField
from pyvale.integratorspatial import (IIntegratorSpatial,
                                           create_int_pt_array)
from pyvale.sensordata import SensorData


#TODO: Docstrings

#NOTE: code below is very similar to quadrature integrator should be able to
# refactor into injected classes/functions

class Rectangle2D(IIntegratorSpatial):
    __slots__ = ("_field","sens_data","_area","_area_int","_n_int_pts",
                 "_int_pt_offsets","_int_pts","_averages")

    def __init__(self,
                 field: IField,
                 sens_data: SensorData,
                 int_pt_offsets: np.ndarray) -> None:

        self._field = field
        self._sens_data = sens_data

        self._area = (self._sens_data.spatial_dims[0]
                      * self._sens_data.spatial_dims[1])
        self._area_int = self._area/int_pt_offsets.shape[0]

        self._n_int_pts = int_pt_offsets.shape[0]
        self._int_pt_offsets = int_pt_offsets
        self._int_pts = create_int_pt_array(self._sens_data,
                                            self._int_pt_offsets)

        self._averages = None


    def calc_integrals(self, sens_data: SensorData | None = None) -> np.ndarray:
        self._averages = self.calc_averages(sens_data)
        return self._area*self.get_averages()


    def get_integrals(self) -> np.ndarray:
        return self._area*self.get_averages()

    def calc_averages(self, sens_data: SensorData | None = None) -> np.ndarray:

        if sens_data is not None:
            self._sens_data = sens_data

        # shape=(n_sens*n_int_pts,n_dims)
        self._int_pts = create_int_pt_array(self._sens_data,
                                            self._int_pt_offsets)


        # shape=(n_int_pts*n_sens,n_comps,n_timesteps)
        int_vals = self._field.sample_field(self._int_pts,
                                            self._sens_data.sample_times,
                                            self._sens_data.angles)

        meas_shape = (self._sens_data.positions.shape[0],
                      int_vals.shape[1],
                      int_vals.shape[2])

        # shape=(n_gauss_pts,n_sens,n_comps,n_timesteps)
        int_vals = int_vals.reshape((self._n_int_pts,)+meas_shape,
                                     order='F')

        # shape=(n_sensors,n_comps,n_timsteps)
        self._averages = 1/self._area * np.sum(self._area_int*int_vals,axis=0)

        return self._averages


    def get_averages(self) -> np.ndarray:
        if self._averages is None:
            self._averages = self.calc_averages()

        return self._averages



