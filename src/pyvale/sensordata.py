# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from pyvale.integratortype import EIntSpatialType


@dataclass(slots=True)
class SensorData:
    """Data class used for specifying sensor array parameters including:
    position, sample times, angles (for vector/tensor fields), spatial averaging
    and spatial dimensions of the sensor for spatial averaging. The number of
    sensor positions specified determines the number of sensors in the array.
    """

    positions: np.ndarray | None = None
    """Numpy array of sensor positions where each row is for an individual
    sensor and the columns specify the X, Y and Z coordinates respectively. To
    create a sensor array the positions must be specified and the number of rows
    of the position array determines the number of sensors in the array.

    shape=(num_sensors,3)
    """

    sample_times: np.ndarray | None = None
    """Numpy array of times at which the sensors will take measurements (sample
    the field). This does not need to be specified to create a sensor array and
    if it is set to None then the sample times will be assumed to be the same as
    the simulation time steps.

    shape=(num_time_steps,)
    """

    angles: tuple[Rotation,...] | None = None
    """The angles for each sensor in the array specified using scipy Rotation
    objects. For scalar fields the rotation only has an effect if a spatial
    averager is specified and the locations of the integration points are
    rotated. For vector and tensor fields the field is transformed using this
    rotation as well as rotating the positions of the integration points if a
    spatial averager is specified.

    Specifying a single rotation in the tuple will cause all sensors to have the
    same rotation and they will be batch processed increasing speed. Otherwise
    this tuple must have a length equal to the number of sensors (i.e. the
    number of rows in the position array above).

    shape=(num_sensor,) | (1,)
    """

    spatial_averager: EIntSpatialType | None = None
    """Type of spatial averaging to use for this sensor array. If None then no
    spatial averaging is performed and sensor values are taken directly from the
    specified positions.
    """

    spatial_dims: np.ndarray | None = None
    """The spatial dimension of the sensor array in its local X,Y,Z coordinates.
    Only used if spatial averager is specified above.

    shape=(3,)
    """





