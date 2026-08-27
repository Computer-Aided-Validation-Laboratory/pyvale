# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Sensor positioning and placement utility functions."""

import numpy as np
from pyvale.sensorsim.sensorarray import ISensorArray


def gen_pos_grid_inside(
    num_sensors: tuple[int, int, int],
    x_lims: tuple[float, float],
    y_lims: tuple[float, float],
    z_lims: tuple[float, float],
) -> np.ndarray:
    """Creates a uniform grid of sensors inside the specified bounds (excluding
    outer boundary edges).

    Parameters
    ----------
    num_sensors : tuple[int, int, int]
        Number of sensors to create in the X, Y, and Z directions.
    x_lims : tuple[float, float]
        Limits of the X axis sensor locations (min, max).
    y_lims : tuple[float, float]
        Limits of the Y axis sensor locations (min, max).
    z_lims : tuple[float, float]
        Limits of the Z axis sensor locations (min, max).

    Returns
    -------
    np.ndarray
        Array of sensor positions with shape (num_sensors, 3).
    """
    sens_pos_x = np.linspace(x_lims[0], x_lims[1], num_sensors[0] + 2)[1:-1]
    sens_pos_y = np.linspace(y_lims[0], y_lims[1], num_sensors[1] + 2)[1:-1]
    sens_pos_z = np.linspace(z_lims[0], z_lims[1], num_sensors[2] + 2)[1:-1]

    sens_grid_x, sens_grid_y, sens_grid_z = np.meshgrid(
        sens_pos_x, sens_pos_y, sens_pos_z
    )

    sens_pos = np.vstack(
        (sens_grid_x.flatten(), sens_grid_y.flatten(), sens_grid_z.flatten())
    ).T
    return sens_pos


def gen_pos_grid_boundary(
    num_sensors: tuple[int, int, int],
    x_lims: tuple[float, float],
    y_lims: tuple[float, float],
    z_lims: tuple[float, float],
) -> np.ndarray:
    """Creates a uniform grid of sensors inclusive of the outer boundary edges.

    Parameters
    ----------
    num_sensors : tuple[int, int, int]
        Number of sensors to create in the X, Y, and Z directions.
    x_lims : tuple[float, float]
        Limits of the X axis sensor locations (min, max).
    y_lims : tuple[float, float]
        Limits of the Y axis sensor locations (min, max).
    z_lims : tuple[float, float]
        Limits of the Z axis sensor locations (min, max).

    Returns
    -------
    np.ndarray
        Array of sensor positions with shape (num_sensors, 3).
    """

    def _axis_linspace(n: int, lims: tuple[float, float]) -> np.ndarray:
        if n <= 1:
            return np.array([(lims[0] + lims[1]) / 2.0], dtype=np.float64)
        return np.linspace(lims[0], lims[1], n)

    sens_pos_x = _axis_linspace(num_sensors[0], x_lims)
    sens_pos_y = _axis_linspace(num_sensors[1], y_lims)
    sens_pos_z = _axis_linspace(num_sensors[2], z_lims)

    sens_grid_x, sens_grid_y, sens_grid_z = np.meshgrid(
        sens_pos_x, sens_pos_y, sens_pos_z
    )

    sens_pos = np.vstack(
        (sens_grid_x.flatten(), sens_grid_y.flatten(), sens_grid_z.flatten())
    ).T
    return sens_pos


def gen_pos_cylinder(
    num_theta: int,
    num_z: int,
    radius: float,
    z_lims: tuple[float, float],
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    endpoint_theta: bool = False,
    endpoint_z: bool = True,
) -> np.ndarray:
    """Generates sensor positions uniformly distributed on a cylindrical
    surface.

    Parameters
    ----------
    num_theta : int
        Number of angular divisions around the cylinder circumference.
    num_z : int
        Number of height divisions along the cylinder axis (Z).
    radius : float
        Cylinder radius.
    z_lims : tuple[float, float]
        Height limits (z_min, z_max) along the cylinder axis.
    center : tuple[float, float, float], optional
        Cylinder center coordinates (default (0, 0, 0)).
    endpoint_theta : bool, optional
        Whether to include 2*pi in theta (default False to avoid overlapping).
    endpoint_z : bool, optional
        Whether to include the top boundary height (default True).

    Returns
    -------
    np.ndarray
        Array of sensor positions with shape (num_theta * num_z, 3).
    """
    thetas = np.linspace(
        0.0, 2.0 * np.pi, num_theta, endpoint=endpoint_theta, dtype=np.float64
    )
    if num_z <= 1:
        z_vals = np.array([(z_lims[0] + z_lims[1]) / 2.0], dtype=np.float64)
    else:
        z_vals = np.linspace(
            z_lims[0], z_lims[1], num_z, endpoint=endpoint_z, dtype=np.float64
        )

    grid_theta, grid_z = np.meshgrid(thetas, z_vals)
    theta_flat = grid_theta.flatten()
    z_flat = grid_z.flatten()

    x_coords = center[0] + radius * np.cos(theta_flat)
    y_coords = center[1] + radius * np.sin(theta_flat)
    z_coords = center[2] + z_flat

    return np.vstack((x_coords, y_coords, z_coords)).T


def gen_pos_sphere(
    num_sensors: int,
    radius: float,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Generates sensor positions uniformly distributed on a spherical surface
    using the Fibonacci equal-area lattice.

    Parameters
    ----------
    num_sensors : int
        Total number of sensors to place on the sphere.
    radius : float
        Sphere radius.
    center : tuple[float, float, float], optional
        Sphere center coordinates (default (0, 0, 0)).

    Returns
    -------
    np.ndarray
        Array of sensor positions with shape (num_sensors, 3).
    """
    if num_sensors <= 0:
        return np.empty((0, 3), dtype=np.float64)
    if num_sensors == 1:
        return np.array(
            [[center[0], center[1], center[2] + radius]], dtype=np.float64
        )

    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(num_sensors, dtype=np.float64)

    # Elevation coordinate from 1 to -1
    y_unit = 1.0 - (indices / float(num_sensors - 1)) * 2.0
    radius_xy = np.sqrt(np.maximum(0.0, 1.0 - y_unit * y_unit))

    theta = golden_angle * indices

    x_unit = np.cos(theta) * radius_xy
    z_unit = np.sin(theta) * radius_xy

    x_coords = center[0] + radius * x_unit
    y_coords = center[1] + radius * y_unit
    z_coords = center[2] + radius * z_unit

    return np.vstack((x_coords, y_coords, z_coords)).T


def print_measurements(
    sens_array: ISensorArray,
    sensors: int | slice,
    components: int | slice,
    time_steps: int | slice,
) -> None:
    """Diagnostic function to print sensor measurements to the console."""
    measurement = sens_array.get_measurements()
    truth = sens_array.get_truth()
    rand_errs = sens_array.get_errors_random()
    sys_errs = sens_array.get_errors_systematic()
    tot_errs = sens_array.get_errors_total()

    meas_slice = measurement[sensors, components, time_steps]
    print(f"measurement.shape = \n    {measurement.shape}")
    print(f"measurement = \n    {meas_slice}")
    print(f"truth = \n    {truth[sensors, components, time_steps]}")

    if rand_errs is not None:
        r_slice = rand_errs[sensors, components, time_steps]
        print(f"random errors = \n    {r_slice}")

    if sys_errs is not None:
        s_slice = sys_errs[sensors, components, time_steps]
        print(f"systematic errors = \n    {s_slice}")

    if tot_errs is not None:
        t_slice = tot_errs[sensors, components, time_steps]
        print(f"total errors = \n    {t_slice}")
