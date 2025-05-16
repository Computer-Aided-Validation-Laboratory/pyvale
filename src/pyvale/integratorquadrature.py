# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from typing import Callable
import numpy as np
from pyvale.field import IField
from pyvale.integratorspatial import (IIntegratorSpatial,
                                      create_int_pt_array)
from pyvale.sensordata import SensorData


class Quadrature2D(IIntegratorSpatial):
    """Gaussian quadrature numerical integrator for spatial averaging in 2D.
    Used to model spatial averaging of sensors over a rectangular area which is
    specified in the SensorData object. Handles sampling of the physical field
    at the integration points and averages them back to a single value per
    sensor location as specified in the SensorData object.

    Implements the `IIntegratorSpatial` interface allowing for interoperability
    of different spatial integration algorithms for modelling sensor averaging.
    """
    __slots__ = ("_field","_area","_gauss_pts_num","_gauss_pt_offsets"
                 ,"_gauss_weight_func","_gauss_pts","_averages","_sens_data")

    def __init__(self,
                 field: IField,
                 sens_data: SensorData,
                 gauss_pt_offsets: np.ndarray,
                 gauss_weight_func: Callable) -> None:
        """
        Parameters
        ----------
        field : IField
            A physical field interface that will be sampled at the integration
            points and averaged back to single value per sensor.
        sens_data : SensorData
            Parameters of the sensor array including the sensor locations,
            sampling times, type of spatial integrator and its dimensions. See
            the `SensorData` dataclass for more details.
        gauss_pt_offsets : np.ndarray
            Offsets from the central location of the integration area with
            shape=(n_gauss_pts,coord[X,Y,Z])
        gauss_weight_func : Callable
            A function that takes the shape of the measurement array as a tuple
            and returns a numpy array of weights for the gaussian integration
            points. The function must return an array with shape=(n_gauss_pts,)
            +meas_shape where meas_shape=(num_sensors,num_field_components,
            num_time_steps)
        """
        self._field = field
        self._sens_data = sens_data
        self._area = self._sens_data.spatial_dims[0] * \
            self._sens_data.spatial_dims[1]

        self._gauss_pts_num = gauss_pt_offsets.shape[0]
        self._gauss_pt_offsets = gauss_pt_offsets
        self._gauss_weight_func = gauss_weight_func

        self._gauss_pts = create_int_pt_array(self._sens_data,
                                              self._gauss_pt_offsets)
        self._averages = None

    def calc_integrals(self, sens_data: SensorData | None = None) -> np.ndarray:
        """Calculates the numerical integrals for each sensor based on the
        specified sensor data and numerical integration options (i.e. geometry
        and integration points).

        Parameters
        ----------
        sens_data : SensorData | None, optional
            Specifies the sensor parameters used to calculate the averages, by
            default None. Is a sensor data object is passed a reference to that
            object is stored by this class and used in later calculations. If
            None then it uses the SensorData object stored by this class.
            Defaults to None.

        Returns
        -------
        np.ndarray
            Array of virtual sensor integrals with shape=(n_sensors,n_comps,
            n_timsteps). Note this is consistent with pyvales measurement array.
        """
        self._averages = self.calc_averages(sens_data)
        return self._area*self.get_averages()

    def get_integrals(self) -> np.ndarray:
        """Gets the most recent calculation of the spatial averages for all
        sensors in the sensor array without performing any new interpolation. If
        the averages have not been calculated they are first calculated and then
        returned.

        Returns
        -------
        np.ndarray
            Array of virtual sensor averages with shape=(n_sensors,n_comps,
            n_timsteps). Note this is consistent with pyvales measurement array.
        """
        return self._area*self.get_averages()

    def calc_averages(self, sens_data: SensorData | None = None) -> np.ndarray:
        """Calculates the spatial averages for each sensor based on the
        specified sensor data and numerical integration options (i.e. geometry
        and integration points).

        Parameters
        ----------
        sens_data : SensorData | None, optional
            Specifies the sensor parameters used to calculate the averages, by
            default None. Is a sensor data object is passed a reference to that
            object is stored by this class and used in later calculations. If
            None then it uses the SensorData object stored by this class.
            Defaults to None.

        Returns
        -------
        np.ndarray
            Array of virtual sensor averages with shape=(n_sensors,n_comps,
            n_timsteps). Note this is consistent with pyvales measurement array.
        """
        if sens_data is not None:
            self._sens_data = sens_data

        # shape=(n_sens*n_gauss_pts,n_dims)
        self._gauss_pts = create_int_pt_array(self._sens_data,
                                              self._gauss_pt_offsets)

        # shape=(n_gauss_pts*n_sens,n_comps,n_timesteps)
        gauss_vals = self._field.sample_field(self._gauss_pts,
                                              self._sens_data.sample_times,
                                              self._sens_data.angles)

        meas_shape = (self._sens_data.positions.shape[0],
                      gauss_vals.shape[1],
                      gauss_vals.shape[2])

        # shape=(n_gauss_pts,n_sens,n_comps,n_timesteps)
        gauss_vals = gauss_vals.reshape((self._gauss_pts_num,)+meas_shape,
                                         order='F')

        # shape=(n_gauss_pts,n_sens,n_comps,n_timesteps)
        gauss_weights = self._gauss_weight_func(meas_shape)

        # NOTE: coeff comes from changing gauss interval from [-1,1] to [a,b] -
        # so (a-b)/2 * (a-b)/2 = sensor_area / 4, then need to divide by the
        # integration area to convert to an average:
        # integrals = self._area/4 * np.sum(gauss_weights*gauss_vals,axis=0)
        # self._averages = (1/self._area)*integrals

        # shape=(n_sensors,n_comps,n_timsteps)=meas_shape
        self._averages = 1/4 * np.sum(gauss_weights*gauss_vals,axis=0)
        return self._averages

    def get_averages(self) -> np.ndarray:
        """Gets the most recent calculation of the spatial averages for all
        sensors in the sensor array without performing any new interpolation. If
        the averages have not been calculated they are first calculated and then
        returned.

        Returns
        -------
        np.ndarray
            Array of virtual sensor averages with shape=(n_sensors,n_comps,
            n_timsteps). Note this is consistent with pyvales measurement array.
        """
        if self._averages is None:
            self._averages = self.calc_averages()

        return self._averages


def create_gauss_weights_2d_4pts(meas_shape: tuple[int,int,int]) -> np.ndarray:
    """Helper function that creates an array of weights for gaussian quadrature
    integration. This function provides the weights for 2D integrator with 4
    integration points.

    Parameters
    ----------
    meas_shape : tuple[int,int,int]
        Shape of the measurement array, shape=(n_sensors,n_field_comps,
        n_time_steps).

    Returns
    -------
    np.ndarray
        Array of gaussian quadrature weights with shape = (n_gauss_pts,n_sensors
        ,n_field_comps,n_time_steps).
    """
    #shape=(4,)+meas_shape
    return np.ones((4,)+meas_shape)


def create_gauss_weights_2d_9pts(meas_shape: tuple[int,int,int]) -> np.ndarray:
    """Helper function that creates an array of weights for gaussian quadrature
    integration. This function provides the weights for 2D integrator with 9
    integration points.

    Parameters
    ----------
    meas_shape : tuple[int,int,int]
        Shape of the measurement array, shape=(n_sensors,n_field_comps,
        n_time_steps).

    Returns
    -------
    np.ndarray
        Array of gaussian quadrature weights with shape = (n_gauss_pts,n_sensors
        ,n_field_comps,n_time_steps).
    """
    # shape=(9,)+meas_shape
    gauss_weights = np.vstack((25/81 * np.ones((4,)+meas_shape),
                               40/81 * np.ones((4,)+meas_shape),
                               64/81 * np.ones((1,)+meas_shape)))
    return gauss_weights

