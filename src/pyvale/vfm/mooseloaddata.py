from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyvista as pv

from pyvale import mooseherder, sensorsim
from pyvale.mooseherder.simdata import SimData
from pyvale.vfm.inputdata import MooseConfig


def load_moose_data(config: MooseConfig) -> tuple[
    npt.NDArray[np.float64],  # x, shape (y, x)
    npt.NDArray[np.float64],  # y, shape (y, x)
    npt.NDArray[np.float64],  # strain, shape (timesteps, components, y, x)
    npt.NDArray[np.float64],  # force, shape (timesteps, components)
    npt.NDArray[np.float64],  # time, shape (timesteps,)
]:
    x, y, strain = _load_moose_strain(config)
    force = _load_moose_force(config)
    time = _load_moose_timesteps(config)

    return (x, y, strain, force, time)


def _load_moose_strain(
    config: MooseConfig,
) -> tuple[
    npt.NDArray[np.float64],  # x_grid, shape (y, x)
    npt.NDArray[np.float64],  # y_grid, shape (y, x)
    npt.NDArray[np.float64],  # strain, shape (timesteps, components, y, x)
]:
    (x_grid, y_grid, grid_data) = _load_sim_data_to_grid(
        config,
        config.strain_component_keys,
    )

    return _enforce_conventions(x_grid, y_grid, grid_data)


def _load_sim_data(config: MooseConfig) -> SimData:
    path = Path(config.exodus_file_path)

    return mooseherder.ExodusLoader(path).load_all_sim_data()


def _load_sim_data_to_grid(
    config: MooseConfig,
    component_keys: tuple[str, ...],
) -> tuple[
    npt.NDArray[np.float64],  # x_grid, shape (x, y, z)
    npt.NDArray[np.float64],  # y_grid, shape (x, y, z)
    npt.NDArray[np.float64],  # grid_data, shape (x, y, z, components, timesteps)
]:
    sim_data = _load_sim_data(config)

    def grid_inner_vec(lower: float, upper: float, num_divs: int) -> np.ndarray:
        step = (upper - lower) / num_divs
        start = lower + (step / 2)
        stop = upper - (step / 2)
        return np.linspace(start, stop, num_divs)

    plate_width = config.width
    plate_height = config.height
    grid_divs = config.grid_divs

    x_vec = grid_inner_vec(plate_width / 2, -plate_width / 2, grid_divs)
    y_vec = (
        grid_inner_vec(plate_height / 2, -plate_height / 2, grid_divs)
        + plate_height / 2
    )
    z_vec = np.full((1,), 0.0, dtype=np.float64)

    (x_grid, y_grid, z_grid) = np.meshgrid(x_vec, y_vec, z_vec, indexing="ij")

    interp_grid = np.stack([x_grid, y_grid, z_grid], axis=0)
    spatial_grid_shape = interp_grid.shape[1:]
    interp_points = interp_grid.reshape(3, -1).T

    pyvista_interp = sensorsim.simdata_to_pyvista_interp(
        sim_data,
        component_keys,
        sensorsim.EDim.TWOD,
    )
    pv_points = pv.PolyData(interp_points)
    sample_data = pv_points.sample(pyvista_interp)

    invalid = ~sample_data["vtkValidPointMask"].astype(bool)

    n_comps = len(component_keys)
    (n_sensors, n_time_steps) = np.array(sample_data[component_keys[0]]).shape
    sample_at_sim_time = np.empty((n_sensors, n_comps, n_time_steps))

    for ii, cc in enumerate(component_keys):
        data_mat = np.array(sample_data[cc])
        data_mat[invalid, :] = np.nan
        sample_at_sim_time[:, ii, :] = data_mat

    # Target: (Nx, Ny, Nz, n_comps, n_time_steps)
    final_shape = spatial_grid_shape + (n_comps, n_time_steps)
    grid_data = sample_at_sim_time.reshape(final_shape)

    return (x_grid, y_grid, grid_data)


# TODO: will we get any Fx data? Currently it gets zerod
# Output shape: (timesteps, components) [Fx, Fy]
def _load_moose_force(config: MooseConfig) -> npt.NDArray[np.float64]:
    sim_data = _load_sim_data(config)
    force = sim_data.glob_vars[config.force_key]
    return np.column_stack((np.zeros_like(force), force))


# Output shape: (timesteps,)
def _load_moose_timesteps(config: MooseConfig) -> npt.NDArray[np.float64]:
    sim_data = _load_sim_data(config)
    return sim_data.time


def _enforce_conventions(
    x_grid: npt.NDArray[np.float64],  # shape (x, y, z)
    y_grid: npt.NDArray[np.float64],  # shape (x, y, z)
    grid_data: npt.NDArray[np.float64],  # shape (x, y, z, components, timesteps)
) -> tuple[
    npt.NDArray[np.float64],  # x_grid, shape (y, x)
    npt.NDArray[np.float64],  # y_grid, shape (y, x)
    npt.NDArray[np.float64],  # grid_data, shape (timesteps, components, y, x)
]:
    # remove redundant z component
    x_grid = x_grid[:, :, 0]  # shape: (x, y)
    y_grid = y_grid[:, :, 0]  # shape: (x, y)
    grid_data = grid_data[:, :, 0, :, :]  # shape: (x, y, components, timesteps)

    # reshape the grid and data to use our conventions
    x_grid = x_grid.transpose(1, 0)  # shape: (y, x)
    y_grid = y_grid.transpose(1, 0)  # shape: (y, x)
    # shape: (timesteps, components, y, x)
    grid_data = grid_data.transpose(3, 2, 1, 0)

    # x increases with column number, is constant in each column, always positive
    x_grid = np.fliplr(x_grid)
    x_grid += np.nanmax(x_grid)
    grid_data = np.flip(grid_data, axis=2)

    # y increases with row number, is constant in each row, always positive
    y_grid = np.flipud(y_grid)
    grid_data = np.flip(grid_data, axis=3)

    return (x_grid, y_grid, grid_data)
