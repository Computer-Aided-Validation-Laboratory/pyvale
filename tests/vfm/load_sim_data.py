from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyvista as pv

from pyvale import mooseherder, sensorsim

# TODO: Currently requires the vfmverif repo to exist as a sibling
#   directory, when we have test data in the pyvale repe this should
#   be updated
PYVALE_ROOT = Path(__file__).resolve().parent.parent.parent
VFMVERIF_ROOT = PYVALE_ROOT.parent / "vfmverif_meshref_1"

PLATE_HEIGHT = 35e-3 # m
PLATE_WIDTH = 25e-3 # m


def load_strain(
    exodus_file_name: str,
    grid_divs: int,
) -> tuple[
    npt.NDArray[np.float64],  # x_grid, shape (y, x)
    npt.NDArray[np.float64],  # y_grid, shape (y, x)
    npt.NDArray[np.float64],  # strain, shape (timesteps, components, y, x)
]:
    component_keys = (
        "strain_xx",
        "strain_yy",
        "strain_xy",
    )

    (x_grid, y_grid, grid_data) = _load_sim_data_to_grid(
        exodus_file_name,
        component_keys,
        grid_divs,
    )

    (x_grid, y_grid, grid_data) = _enforce_conventions(
        x_grid,
        y_grid,
        grid_data
    )

    return (
        x_grid,
        y_grid,
        grid_data
    )


def load_stress(
    exodus_file_name: str,
    grid_divs: int,
) -> tuple[
    npt.NDArray[np.float64],  # x_grid, shape (y, x)
    npt.NDArray[np.float64],  # y_grid, shape (y, x)
    npt.NDArray[np.float64],  # stress, MPa, shape (timesteps, components, y, x)
]:
    component_keys = (
        "stress_xx",
        "stress_yy",
        "stress_xy",
    )

    (x_grid, y_grid, grid_data) = _load_sim_data_to_grid(
        exodus_file_name,
        component_keys,
        grid_divs,
    )

    (x_grid, y_grid, grid_data) = _enforce_conventions(
        x_grid,
        y_grid,
        grid_data
    )


    # Stress should be in MPa, convert from Pa to MPa
    grid_data *= 1e-6

    return (
        x_grid,
        y_grid,
        grid_data
    )

# Output shape: (timesteps, components) [Fx, Fy]
def load_force(
    exodus_file_name: str
) -> npt.NDArray[np.float64]:
    exodus_file_path = VFMVERIF_ROOT / exodus_file_name
    sim_data = mooseherder.ExodusLoader(exodus_file_path).load_all_sim_data()
    force = sim_data.glob_vars["react_y_top"]
    return np.column_stack((np.zeros_like(force), force))

# Output shape: (timesteps)
def load_timesteps(
    exodus_file_name: str
) -> npt.NDArray[np.float64]:
    exodus_file_path = VFMVERIF_ROOT / exodus_file_name
    sim_data = mooseherder.ExodusLoader(exodus_file_path).load_all_sim_data()
    return sim_data.time


def _load_sim_data_to_grid(
    exodus_file_name: str,
    component_keys: tuple[str, ...],
    grid_divs: int,
) -> tuple[
    npt.NDArray[np.float64],  # x_grid, shape (x, y, z)
    npt.NDArray[np.float64],  # y_grid, shape (x, y, z)
    npt.NDArray[np.float64],  # grid_data, shape (x, y, z, components, timesteps)
]:
    exodus_file_path = VFMVERIF_ROOT / exodus_file_name

    sim_data = mooseherder.ExodusLoader(exodus_file_path).load_all_sim_data()

    def grid_inner_vec(lower: float, upper: float, num_divs: int) -> np.ndarray:
        step = (upper - lower) / num_divs
        start = lower + (step / 2)
        stop = upper - (step / 2)
        return np.linspace(start, stop, num_divs)

    x_vec = grid_inner_vec(PLATE_WIDTH / 2, -PLATE_WIDTH / 2, grid_divs)
    y_vec = (
        grid_inner_vec(PLATE_HEIGHT / 2, -PLATE_HEIGHT / 2, grid_divs)
        + PLATE_HEIGHT / 2
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

    return (
        x_grid,
        y_grid,
        grid_data,
    )


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
    grid_data = grid_data.transpose(3, 2, 1, 0)  # shape: (timesteps, components, y, x)

    # x increases with column number, is constant in each column, always positive
    x_grid = np.fliplr(x_grid)
    x_grid += np.nanmax(x_grid)
    grid_data = np.flip(grid_data, axis=2)

    # y increases with row number, is constant in each row, always positive
    y_grid = np.flipud(y_grid)
    grid_data = np.flip(grid_data, axis=3)

    return (
        x_grid,
        y_grid,
        grid_data
    )
