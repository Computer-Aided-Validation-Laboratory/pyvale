# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np
from pyvale.field import IField
from pyvale.integratorspatial import (IIntegratorSpatial,
                                           create_int_pt_array)
from pyvale.sensordata import SensorData

#NOTE: code below is very similar to quadrature integrator should be able to
# refactor into injected classes/functions

class Rectangle2D(IIntegratorSpatial):
    """Rectangular numerical integrator for spatial averaging in 2D. Used to
    model spatial averaging of sensors over a rectangular area which is
    specified in the SensorData object. Handles sampling of the physical field
    at the integration points and averages them back to a single value per
    sensor location as specified in the SensorData object.

    Implements the `IIntegratorSpatial` interface allowing for interoperability
    of different spatial integration algorithms for modelling sensor averaging.
    """
    __slots__ = ("_field","sens_data","_area","_area_int","_n_int_pts",
                 "_int_pt_offsets","_int_pts","_averages")

    def __init__(self,
                 field: IField,
                 sens_data: SensorData,
                 int_pt_offsets: np.ndarray) -> None:
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
        int_pt_offsets : np.ndarray
            Offsets from the central location of the integration area with
            shape=(n_gauss_pts,coord[X,Y,Z])
        """
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



