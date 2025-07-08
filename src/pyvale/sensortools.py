# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================
import numpy as np
import mooseherder as mh
from pyvale.sensorarray import ISensorArray


def create_sensor_pos_array(num_sensors: tuple[int,int,int],
                           x_lims: tuple[float, float],
                           y_lims: tuple[float, float],
                           z_lims: tuple[float, float]) -> np.ndarray:
    """Function or creating a uniform grid of sensors inside the specified
    bounds and returning the positions in format that can be used to build a
    `SensorData` object.

    To create a line of sensors along the X axis set the number of sensors to 1
    for all the Y and Z axes and then set the upper and lower limits of the Y
    and Z axis to be the same value.

    To create a plane of sensors in the X-Y plane set the number of sensors in
    Z to 1 and set the upper and lower coordinates of the Z limit to the desired
    Z location of the plane. Then set the number of sensors in X and Y as
    desired along with the associated limits.

    Parameters
    ----------
    n_sens : tuple[int,int,int]
        Number of sensors to create in the X, Y and Z directions.
    x_lims : tuple[float, float]
        Limits of the X axis sensor locations.
    y_lims : tuple[float, float]
        Limits of the Y axis sensor locations.
    z_lims : tuple[float, float]
        Limits of the Z axis sensor locations.

    Returns
    -------
    np.ndarray
        Array of sensor positions with shape=(num_sensors,3) where num_sensors
        is the product of integers in the num_sensors tuple. The columns are the
        X, Y and Z locations of the sensors.
    """
    sens_pos_x = np.linspace(x_lims[0],x_lims[1],num_sensors[0]+2)[1:-1]
    sens_pos_y = np.linspace(y_lims[0],y_lims[1],num_sensors[1]+2)[1:-1]
    sens_pos_z = np.linspace(z_lims[0],z_lims[1],num_sensors[2]+2)[1:-1]

    (sens_grid_x,sens_grid_y,sens_grid_z) = np.meshgrid(
        sens_pos_x,sens_pos_y,sens_pos_z)

    sens_pos_x = sens_grid_x.flatten()
    sens_pos_y = sens_grid_y.flatten()
    sens_pos_z = sens_grid_z.flatten()

    sens_pos = np.vstack((sens_pos_x,sens_pos_y,sens_pos_z)).T
    return sens_pos


def print_measurements(sens_array: ISensorArray,
                       sensors: int | slice,
                       components: int | slice,
                       time_steps: int | slice)  -> None:
    """Diagnostic function to print sensor measurements to the console. Also
    prints the ground truth, the random and the systematic errors for the
    specified sensor array. The sensors, components and time steps are specified
    as slices of the measurement array.

    Parameters
    ----------
    sens_array : ISensorArray
        Sensor array to print measurements for.
    sensors : int | slice
        Index for the sensor or slice of range of sensors to be printed to the
        console.
    components : int | slice
        Index for the field component or slice of range of field components to
        be printed to the console
    time_steps : int | slice
        Index for the time step or slice of time steps to be printed to the
        console.
    """

    measurement =  sens_array.get_measurements()
    truth = sens_array.get_truth()
    rand_errs = sens_array.get_errors_random()
    sys_errs = sens_array.get_errors_systematic()
    tot_errs = sens_array.get_errors_total()

    print(f"\nmeasurement.shape = \n    {measurement.shape}")
    print(f"measurement = \n    {measurement[sensors,components,time_steps]}")
    print(f"truth = \n    {truth[sensors,components,time_steps]}")

    if rand_errs is not None:
        print(f"random errors = \n    {rand_errs[sensors,components,time_steps]}")

    if sys_errs is not None:
        print(f"systematic errors = \n    {sys_errs[sensors,components,time_steps]}")

    if tot_errs is not None:
        print(f"total errors = \n    {tot_errs[sensors,components,time_steps]}")

    print()


def print_dimensions(sim_data: mh.SimData) -> None:
    """Diagnostic function for quickly finding the coordinate limits for from a
    given simulation.

    Parameters
    ----------
    sim_data : mh.SimData
        Simulation data objects containing the nodal coordinates.
    """
    print(80*"-")
    print("SimData Dimensions:")
    print(f"x [min,max] = [{np.min(sim_data.coords[:,0])}," + \
          f"{np.max(sim_data.coords[:,0])}]")
    print(f"y [min,max] = [{np.min(sim_data.coords[:,1])}," + \
          f"{np.max(sim_data.coords[:,1])}]")
    print(f"z [min,max] = [{np.min(sim_data.coords[:,2])}," + \
          f"{np.max(sim_data.coords[:,2])}]")
    print(f"t [min,max] = [{np.min(sim_data.time)},{np.max(sim_data.time)}]")
    print(80*"-")

