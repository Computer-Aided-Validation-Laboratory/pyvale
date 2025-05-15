# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np
import pyvista as pv
from pyvale.field import IField
from pyvale.sensordata import SensorData
from pyvale.integratorfactory import build_spatial_averager


def sample_field_with_sensor_data(field: IField, sensor_data: SensorData
                                  ) -> np.ndarray:
    """Samples (interpolates) an `IField` object using the parameters specified
    in the `SensorData` object.

    Parameters
    ----------
    field : IField
        The simulated physical field that the sensors will samples from. This is
        normally a `FieldScalar`, `FieldVector` or `FieldTensor`.
    sensor_data : SensorData
        Contains sensor array parameters including: number of sensors, positions
        and sample times. See the `SensorData` class for more information.

    Returns
    -------
    np.ndarray
        Array of sampled sensor measurements with shape=(num_sensors,
        num_field_components,num_time_steps).
    """
    if sensor_data.spatial_averager is None:
        return field.sample_field(sensor_data.positions,
                                  sensor_data.sample_times,
                                  sensor_data.angles)

    spatial_integrator = build_spatial_averager(field,sensor_data)
    return spatial_integrator.calc_averages()


# NOTE: sampling outside the bounds of the sample returns a value of 0
def sample_pyvista_grid(components: tuple[str,...],
                        pyvista_grid: pv.UnstructuredGrid,
                        sim_time_steps: np.ndarray,
                        points: np.ndarray,
                        sample_times: np.ndarray | None = None
                        ) -> np.ndarray:
    """Function for sampling (interpolating) a pyvista grid object containing
    simulated field data. The pyvista sample method uses VTK to perform the
    spatial interpolation using the element shape functions. If the sampling
    time steps are not the same as the simulation time then a linear
    interpolation over time is performed using numpy.

    NOTE: sampling outside the mesh bounds of the sample returns a value of 0.

    Parameters
    ----------
    components : tuple[str,...]
        String keys for the components to be sampled in the pyvista grid object.
        Useful for only interpolating the field components of interest for speed
        and memory reduction.
    pyvista_grid : pv.UnstructuredGrid
        Pyvista grid object containing the simulation mesh and the components of
        the physical field that will be sampled.
    sim_time_steps : np.ndarray
        Simulation time steps corresponding to the fields in the pyvista grid
        object.
    points : np.ndarray
        Coordinates of the points at which to sample the pyvista grid object.
        shape=(num_points,3) where the columns are the X, Y and Z coordinates of
        the sample points in simulation world coordintes.
    sample_times : np.ndarray | None, optional
        Array of time steps at which to sample the pyvista grid. If None then no
        temporal interpolation is performed and the sample times are assumed to
        be the simulation time steps.

    Returns
    -------
    np.ndarray
        Array of sampled sensor measurements with shape=(num_sensors,
        num_field_components,num_time_steps).
    """
    pv_points = pv.PolyData(points)
    sample_data = pv_points.sample(pyvista_grid)

    n_comps = len(components)
    (n_sensors,n_time_steps) = np.array(sample_data[components[0]]).shape
    sample_at_sim_time = np.empty((n_sensors,n_comps,n_time_steps))

    for ii,cc in enumerate(components):
        sample_at_sim_time[:,ii,:] = np.array(sample_data[cc])

    if sample_times is None:
        return sample_at_sim_time

    def sample_time_interp(x):
        return np.interp(sample_times, sim_time_steps, x)

    n_time_steps = sample_times.shape[0]
    sample_at_spec_time = np.empty((n_sensors,n_comps,n_time_steps))

    for ii,cc in enumerate(components):
        sample_at_spec_time[:,ii,:] = np.apply_along_axis(sample_time_interp,-1,
                                                    sample_at_sim_time[:,ii,:])

    return sample_at_spec_time



