from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator


GMSH_HEX20 = 17
GMSH_VOLUME_DIM = 3


def read_gmsh_nodes(lines: list[str]) -> np.ndarray:
    start = lines.index("$Nodes") + 1
    num_entity_blocks, num_nodes, _, _ = map(int, lines[start].split())
    node_coords: dict[int, np.ndarray] = {}

    line_idx = start + 1
    for _ in range(num_entity_blocks):
        _, _, _, num_nodes_in_block = map(int, lines[line_idx].split())
        line_idx += 1

        node_tags = [int(lines[line_idx + ii]) for ii in range(num_nodes_in_block)]
        line_idx += num_nodes_in_block

        for tag in node_tags:
            node_coords[tag] = np.fromstring(lines[line_idx], sep=" ", dtype=np.float64)
            line_idx += 1

    sorted_tags = sorted(node_coords)
    if len(sorted_tags) != num_nodes:
        raise ValueError(f"Expected {num_nodes} nodes, found {len(sorted_tags)}.")

    nodes = np.empty((sorted_tags[-1] + 1, 3), dtype=np.float64)
    for tag in sorted_tags:
        nodes[tag] = node_coords[tag]

    return nodes


def read_gmsh_element_connectivity(
    lines: list[str],
    entity_dim: int = GMSH_VOLUME_DIM,
    element_type: int = GMSH_HEX20,
) -> np.ndarray:
    start = lines.index("$Elements") + 1
    num_entity_blocks, _, _, _ = map(int, lines[start].split())
    connectivity: list[list[int]] = []

    line_idx = start + 1
    for _ in range(num_entity_blocks):
        block_entity_dim, _, block_element_type, num_in_block = map(int, lines[line_idx].split())
        line_idx += 1

        for _ in range(num_in_block):
            entry = [int(value) for value in lines[line_idx].split()]
            line_idx += 1

            if block_entity_dim == entity_dim and block_element_type == element_type:
                connectivity.append(entry[1:])

    if not connectivity:
        raise ValueError(
            f"No elements found for entity_dim={entity_dim}, element_type={element_type}."
        )

    return np.asarray(connectivity, dtype=np.int64)


def read_gmsh_element_centres(
    mesh_path: Path,
    entity_dim: int = GMSH_VOLUME_DIM,
    element_type: int = GMSH_HEX20,
) -> np.ndarray:
    lines = mesh_path.read_text().splitlines()
    nodes = read_gmsh_nodes(lines)
    connectivity = read_gmsh_element_connectivity(lines, entity_dim, element_type)
    return nodes[connectivity].mean(axis=1)


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
    mesh_path: Path,
    element_values_path: Path,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    value_column: int = 0,
    value_scale: float = 1.0,
    specimen_mask: np.ndarray | None = None,
    entity_dim: int = GMSH_VOLUME_DIM,
    element_type: int = GMSH_HEX20,
) -> tuple[np.ndarray, np.ndarray]:
    element_centres = read_gmsh_element_centres(
        mesh_path,
        entity_dim=entity_dim,
        element_type=element_type,
    )
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
