# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from abc import ABC, abstractmethod
import numpy as np
from pyvale.sensordata import SensorData


def create_int_pt_array(sens_data: SensorData,
                        int_pt_offsets: np.ndarray,
                        ) -> np.ndarray:
    """Creates the integration point locations in local coordinates based on the
    specified offsets from the local origin.

    Parameters
    ----------
    sens_data : SensorData
        Contains the parameters of the sensor array including: positions, sample
        times and orientations. If specified the sensor orientations are used
        to rotate the positions of the integration points.
    int_pt_offsets : np.ndarray
        Offsets of the intergation points in non-rotated local coordinates.

    Returns
    -------
    np.ndarray
        The integration point locations in world (simulation) coordinates. The
        rows of the array are all the integration points for all sensors and the
        columns are the X,Y,Z coordinates. shape=(num_sensors*num_int_points,3).
    """
    n_sens = sens_data.positions.shape[0]
    n_int_pts = int_pt_offsets.shape[0]

    # shape=(n_sens*n_int_pts,n_dims)
    offset_array = np.tile(int_pt_offsets,(n_sens,1))

    if sens_data.angles is not None:
        for ii,rr in enumerate(sens_data.angles):
            offset_array[ii*n_int_pts:(ii+1)*n_int_pts,:] = \
                np.matmul(rr.as_matrix(),int_pt_offsets.T).T

    # shape=(n_sens*n_int_pts,n_dims)
    int_pt_array = np.repeat(sens_data.positions,int_pt_offsets.shape[0],axis=0)

    return int_pt_array + offset_array


class IIntegratorSpatial(ABC):
    """Interface (abstract base class) for spatial integrators. Used for
    averaging sensor values over a given space.
    """

    @abstractmethod
    def calc_averages(self, sens_data: SensorData) -> np.ndarray:
        """Abstract method. Calculates the spatial average for each sensor using
        the specified sensor dimensions and integration method. This is done by
        interpolating the sensor values at each sensors integration points.

        Parameters
        ----------
        sens_data : SensorData
            Contains the parameters of the sensor array including: positions,
            sample times, spatial averaging and orientations.

        Returns
        -------
        np.ndarray
            Array of simulated sensor measurements. shape=(num_sensors,
            num_field_components,num_time_steps).
        """


    @abstractmethod
    def get_averages(self) -> np.ndarray:
        """Abstract method. Returns the previously calculated spatial averages
        for each sensor. If these have not been calculated then `calc_averages`
        is called and the result is returned.

        Returns
        -------
        np.ndarray
            Array of simulated sensor measurements. shape=(num_sensors,
            num_field_components,num_time_steps).
        """


