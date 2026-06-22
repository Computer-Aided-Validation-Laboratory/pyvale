from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from pyvale import mooseherder


def _normalise_connectivity(connectivity: np.ndarray) -> np.ndarray:
    connectivity = np.asarray(connectivity, dtype=np.int64)

    if connectivity.ndim != 2:
        raise ValueError(
            f"Expected a 2D connectivity table, got shape {connectivity.shape}."
        )

    if connectivity.shape[0] > connectivity.shape[1]:
        return connectivity

    if connectivity.shape[1] > connectivity.shape[0]:
        return connectivity.T

    raise ValueError(
        f"Could not determine connectivity orientation for square shape {connectivity.shape}."
    )


def _get_connectivity_for_block(
    connect: dict[str, np.ndarray] | None,
    block_num: int,
) -> np.ndarray:
    if not connect:
        raise ValueError("No Exodus connectivity tables were found.")

    connect_key = f"connect{block_num}"
    if connect_key not in connect:
        available = ", ".join(str(key) for key in connect)
        raise KeyError(
            f"Connectivity table '{connect_key}' not found. Available tables: {available}."
        )

    return np.asarray(connect[connect_key], dtype=np.int64)


def read_exodus_element_centres(
    exodus_path: Path,
    *,
    block_num: int = 1,
) -> np.ndarray:
    sim_data = mooseherder.ExodusLoader(exodus_path).load_all_sim_data()

    if sim_data.coords is None:
        raise ValueError("No nodal coordinates were found in the Exodus file.")

    connectivity = _get_connectivity_for_block(sim_data.connect, block_num)
    connectivity = _normalise_connectivity(connectivity)

    if np.min(connectivity) < 1:
        raise ValueError("Expected Exodus connectivity to use 1-based node indices.")

    zero_based_connectivity = connectivity - 1
    return sim_data.coords[zero_based_connectivity].mean(axis=1)


def prepare_element_values(
    element_values: np.ndarray,
    value_column: int = 0,
) -> np.ndarray:
    if element_values.ndim == 2:
        return np.asarray(element_values[:, value_column], dtype=np.float64)
    return np.asarray(element_values, dtype=np.float64)


def interpolate_fe_elements_to_grid(
    element_centres: np.ndarray,
    element_values: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    value_scale: float = 1.0,
    specimen_mask: np.ndarray | None = None,
) -> np.ndarray:
    flat_values = prepare_element_values(element_values) * value_scale

    if flat_values.shape[0] != element_centres.shape[0]:
        raise ValueError(
            f"Element data length {flat_values.shape[0]} does not match "
            f"mesh cell count {element_centres.shape[0]}."
        )

    interpolator = LinearNDInterpolator(element_centres[:, :2], flat_values)
    grid_values = interpolator(x_grid, y_grid)

    if specimen_mask is not None:
        grid_values = np.asarray(grid_values, dtype=np.float64)
        grid_values[~specimen_mask] = np.nan

    return grid_values


def load_and_interpolate_fe_elements(
    exodus_path: Path,
    element_values_path: Path,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    value_column: int = 0,
    value_scale: float = 1.0,
    specimen_mask: np.ndarray | None = None,
    block_num: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    element_centres = read_exodus_element_centres(exodus_path, block_num=block_num)
    element_values = np.load(element_values_path)
    grid_values = interpolate_fe_elements_to_grid(
        element_centres,
        element_values[:, value_column] if np.asarray(element_values).ndim == 2 else element_values,
        x_grid,
        y_grid,
        value_scale=value_scale,
        specimen_mask=specimen_mask,
    )
    return grid_values, element_centres


def plot_interpolated_grid(
    grid_values: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    title: str,
    colorbar_label: str,
    output_path: Path | None = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    image = ax.imshow(
        grid_values,
        extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
        origin="lower",
        aspect="equal",
        cmap="viridis",
    )
    fig.colorbar(image, ax=ax, label=colorbar_label)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    if output_path is not None:
        fig.savefig(output_path, dpi=200)

    plt.show()
