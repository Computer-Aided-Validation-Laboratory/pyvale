from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyvista as pv

from pyvale import mooseherder, sensorsim

PYVALE_ROOT = Path(__file__).resolve().parent.parent.parent
VFMVERIF_ROOT = PYVALE_ROOT.parent / "vfmverif_meshref_1"

PLATE_HEIGHT = 35e-3    # m
PLATE_WIDTH = 25e-3     # m


def _load_sim_data_to_grid(
    exodus_file_name: str,
    component_keys: tuple[str, ...],
    grid_divs: int,
) -> tuple[
    npt.NDArray[np.float64],  # x_grid, shape (x, y, z)
    npt.NDArray[np.float64],  # y_grid, shape (x, y, z)
    npt.NDArray[np.float64],  # grid_data, shape (x, y, z, components, timesteps)
    npt.NDArray[np.float64],  # force, shape (timesteps)
    npt.NDArray[np.float64],  # time, shape (timesteps)
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
        sim_data.glob_vars["react_y_top"],
        sim_data.time,
    )
