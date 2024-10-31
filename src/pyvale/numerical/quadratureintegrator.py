'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
from typing import Callable

import numpy as np

from pyvale.physics.field import IField
from pyvale.numerical.spatialintegrator import (ISpatialAverager,
                                                create_int_pt_array)


def create_gauss_weights_2d_4pts(meas_shape: tuple[int,int,int]) -> np.ndarray:
    return np.ones((4,)+meas_shape)


def create_gauss_weights_2d_9pts(meas_shape: tuple[int,int,int]) -> np.ndarray:
    # shape=(9,)+meas_shape
    gauss_weights = np.vstack((25/81 * np.ones((4,)+meas_shape),
                            40/81 * np.ones((4,)+meas_shape),
                            64/81 * np.ones((1,)+meas_shape)))
    return gauss_weights


class Quadrature2D(ISpatialAverager):
    __slots__ = ("_field","_cent_pos","_area_dims","_sample_times","_area",
                 "_n_gauss_pts","_gauss_pt_offsets","_gauss_weight_func",
                 "_gauss_pts","_integrals")

    def __init__(self,
                 gauss_pt_offsets: np.ndarray,
                 gauss_weight_func: Callable,
                 field: IField,
                 cent_pos: np.ndarray,
                 area_dims: np.ndarray,
                 sample_times: np.ndarray | None = None) -> None:

        self._field = field
        self._cent_pos = cent_pos
        self._area_dims = area_dims
        self._sample_times = sample_times

        self._area = self._area_dims[0]*self._area_dims[1]

        self._n_gauss_pts = gauss_pt_offsets.shape[0]
        self._gauss_pt_offsets = gauss_pt_offsets
        self._gauss_weight_func = gauss_weight_func

        self._gauss_pts = create_int_pt_array(self._gauss_pt_offsets,
                                              cent_pos)

        self._integrals = self.calc_integrals(None, sample_times)


    def calc_integrals(self,
                       cent_pos: np.ndarray | None = None,
                       sample_times: np.ndarray | None = None) -> np.ndarray:

        if cent_pos is not None:
            # shape=(n_sens*n_gauss_pts,n_dims)
            self._gauss_pts = create_int_pt_array(self._gauss_pt_offsets,
                                                  cent_pos)

        # shape=(n_gauss_pts*n_sens,n_comps,n_timesteps)
        gauss_vals = self._field.sample_field(self._gauss_pts,
                                              sample_times)

        meas_shape = (self._cent_pos.shape[0],
                        gauss_vals.shape[1],
                        gauss_vals.shape[2])

        # shape=(n_gauss_pts,n_sens,n_comps,n_timesteps)
        gauss_vals = gauss_vals.reshape((self._n_gauss_pts,)+meas_shape,
                                         order='F')

        # shape=(n_gauss_pts,n_sens,n_comps,n_timesteps)
        gauss_weights = self._gauss_weight_func(meas_shape)

        # shape=(n_sensors,n_comps,n_timsteps)
        # NOTE: coeff comes from changing gauss interval from [-1,1] to [a,b] -
        # so (a-b)/2 * (a-b)/2 = sensor_area / 4, then need to divide by the
        # integration area to convert to an average.
        self._integrals = self._area/4 * np.sum(gauss_weights*gauss_vals,axis=0)

        return self._integrals

    def get_integrals(self) -> np.ndarray:
        return self._integrals

    def calc_averages(self,
                      cent_pos: np.ndarray | None = None,
                      sample_times: np.ndarray | None = None) -> np.ndarray:
        return (1/self._area)*self.calc_integrals(cent_pos,sample_times)

    def get_averages(self) -> np.ndarray:
        return (1/self._area)*self._integrals

