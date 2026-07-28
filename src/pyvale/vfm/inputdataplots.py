from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pyvale.vfm.experimentdata import (
    Edge,
    EdgeConditions,
    EEdgeCondition,
)

# Strain components are stored in [xx, yy, xy] order (see load_ansys_data /
# load_moose_data), so the plot titles follow the same order.
_STRAIN_COMPONENT_NAMES: tuple[str, ...] = (
    "strain_xx",
    "strain_yy",
    "strain_xy"
)

# Force columns are ordered [Fx, Fy, ...]; used to label the force plots.
_FORCE_COMPONENT_LABELS: tuple[str, ...] = ("Fx", "Fy", "Fz")

def _create_diagnostic_plots(
    output_folder: Path,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    strain: npt.NDArray[np.float64],
    force: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    edge_conditions: EdgeConditions,
    component_names: tuple[str, ...] = _STRAIN_COMPONENT_NAMES,
) -> dict[str, Path]:
    """Build the input-data diagnostic plots and save them to ``output_folder``.

    Returns a map of plot name -> saved file path.
    """
    # Points inside the specimen have finite strain; outside, the interpolation
    # leaves NaNs. That finite region doubles as the specimen mask.
    specimen_mask = np.isfinite(strain[0, 0])

    figures: dict[str, Figure] = {
        "coordinate_fields": _plot_coordinate_fields(x, y, specimen_mask),
        "force_time_checks": _plot_force_and_time(force, time),
        "strain_component_checks": _plot_strain_components(
            x, y, strain, specimen_mask, component_names
        ),
        "boundary_conditions": _plot_boundary_conditions(
            x, y, specimen_mask, force, edge_conditions
        ),
    }

    output_folder.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}
    for name, figure in figures.items():
        path = output_folder / f"{name}.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        saved_paths[name] = path

    return saved_paths


def _plot_coordinate_fields(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    specimen_mask: npt.NDArray[np.bool_],
) -> Figure:
    fig, (ax_x, ax_y, ax_mask) = plt.subplots(1, 3, figsize=(15, 4.5))
    _scatter_field(ax_x, x, y, x, "x coordinates")
    _scatter_field(ax_y, x, y, y, "y coordinates")
    ax_mask.imshow(specimen_mask, cmap="gray", interpolation="nearest")
    ax_mask.set(title="Specimen mask", xlabel="x index", ylabel="y index")
    fig.tight_layout()
    return fig


def _plot_force_and_time(
    force: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
) -> Figure:
    fig, (ax_time, ax_force_step, ax_force_time) = plt.subplots(
        1, 3, figsize=(15, 4.5)
    )
    steps = np.arange(time.size)
    labels = _force_component_labels(force.shape[1])

    ax_time.plot(steps, time, marker="o")
    ax_time.set(title="Time by timestep", xlabel="Timestep index", ylabel="Time")
    ax_time.grid(True, alpha=0.3)

    for index, label in enumerate(labels):
        ax_force_step.plot(steps, force[:, index], marker="o", label=label)
    ax_force_step.set(
        title="Force by timestep", xlabel="Timestep index", ylabel="Force"
    )
    ax_force_step.grid(True, alpha=0.3)
    ax_force_step.legend()

    for index, label in enumerate(labels):
        ax_force_time.plot(time, force[:, index], marker="o", label=label)
    ax_force_time.set(title="Force vs time", xlabel="Time", ylabel="Force")
    ax_force_time.grid(True, alpha=0.3)
    ax_force_time.legend()

    fig.tight_layout()
    return fig


def _plot_strain_components(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    strain: npt.NDArray[np.float64],
    specimen_mask: npt.NDArray[np.bool_],
    component_names: tuple[str, ...],
) -> Figure:
    component_count = min(strain.shape[1], len(component_names))
    fig, axes = plt.subplots(
        2, component_count, figsize=(5 * component_count, 8), squeeze=False
    )
    for index in range(component_count):
        name = component_names[index]
        _scatter_field(
            axes[0, index], x, y, strain[0, index],
            f"{name} at first timestep", valid_mask=specimen_mask,
        )
        _scatter_field(
            axes[1, index], x, y, strain[-1, index],
            f"{name} at last timestep", valid_mask=specimen_mask,
        )
    fig.tight_layout()
    return fig


def _scatter_field(
    ax: Axes,
    x_coords: npt.NDArray[np.float64],
    y_coords: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    title: str,
    valid_mask: npt.NDArray[np.bool_] | None = None,
) -> None:
    """Scatter ``values`` at their physical coordinates, skipping NaNs."""
    ax.set(title=title, xlabel="x", ylabel="y")

    valid = np.isfinite(x_coords) & np.isfinite(y_coords) & np.isfinite(values)
    if valid_mask is not None:
        valid &= valid_mask
    if not valid.any():
        return

    scatter = ax.scatter(
        x_coords[valid], y_coords[valid], c=values[valid],
        s=6, linewidths=0.0, cmap="viridis",
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()  # image convention: increasing y downwards
    ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)


def _plot_boundary_conditions(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    specimen_mask: npt.NDArray[np.bool_],
    force: npt.NDArray[np.float64],
    edge_conditions: EdgeConditions,
) -> Figure:
    valid = specimen_mask & np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        raise ValueError(
            "Could not plot boundary conditions because the specimen mask "
            "is empty."
        )

    x_valid = x[valid]
    y_valid = y[valid]
    x_min, x_max = float(x_valid.min()), float(x_valid.max())
    y_min, y_max = float(y_valid.min()), float(y_valid.max())
    # Coordinates are physical (metre-scale), so size annotations off the actual
    # specimen extent rather than an absolute floor.
    scale = max(x_max - x_min, y_max - y_min) or 1.0

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        x_valid, y_valid, s=3, c="#c7d2da", alpha=0.6,
        linewidths=0.0, label="Specimen points",
    )
    ax.set(xlabel="Physical x", ylabel="Physical y",
           title="Boundary-condition diagnostic")
    ax.set_aspect("equal")

    edge_segments = {
        "min_x_edge": ((x_min, y_min), (x_min, y_max), edge_conditions.min_x_edge),
        "max_x_edge": ((x_max, y_min), (x_max, y_max), edge_conditions.max_x_edge),
        "min_y_edge": ((x_min, y_min), (x_max, y_min), edge_conditions.min_y_edge),
        "max_y_edge": ((x_min, y_max), (x_max, y_max), edge_conditions.max_y_edge),
    }
    for edge_name, (start, end, edge) in edge_segments.items():
        color = _edge_boundary_colour(edge)
        ax.plot(
            [start[0], end[0]], [start[1], end[1]],
            color=color, linewidth=3.0, solid_capstyle="round",
        )
        ax.text(
            0.5 * (start[0] + end[0]), 0.5 * (start[1] + end[1]),
            _edge_boundary_label(edge_name, edge),
            fontsize=8, color=color, ha="center", va="center",
            bbox={"facecolor": "white", "alpha": 0.8,
                  "edgecolor": "none", "pad": 1.0},
        )

    _annotate_force_arrow(ax, force, edge_conditions, scale,
                          (x_min, x_max, y_min, y_max))

    margin = 0.25 * scale
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_max + margin, y_min - margin)  # inverted: image convention
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    return fig


def _annotate_force_arrow(
    ax: Axes,
    force: npt.NDArray[np.float64],
    edge_conditions: EdgeConditions,
    scale: float,
    bounds: tuple[float, float, float, float],
) -> None:
    """Draw the applied-force arrow on the traction edge, if there is one."""
    traction_edge_name = _find_traction_edge_name(edge_conditions)
    if traction_edge_name is None:
        return

    x_min, x_max, y_min, y_max = bounds
    arrow_origins = {
        "min_x_edge": (x_min, 0.5 * (y_min + y_max)),
        "max_x_edge": (x_max, 0.5 * (y_min + y_max)),
        "min_y_edge": (0.5 * (x_min + x_max), y_min),
        "max_y_edge": (0.5 * (x_min + x_max), y_max),
    }
    origin_x, origin_y = arrow_origins[traction_edge_name]
    vector_x, vector_y = _resolve_force_arrow_vector(force, traction_edge_name)
    length = 0.18 * scale

    ax.arrow(
        origin_x, origin_y, length * vector_x, length * vector_y,
        color="tab:orange", width=0.01 * scale, length_includes_head=True,
        head_width=0.04 * scale, head_length=0.06 * scale,
    )
    ax.text(
        origin_x + 0.55 * length * vector_x,
        origin_y + 0.55 * length * vector_y,
        "Applied force", color="tab:orange", fontsize=9,
        ha="left" if vector_x >= 0.0 else "right",
        va="bottom" if vector_y <= 0.0 else "top",
    )


def _force_component_labels(component_count: int) -> tuple[str, ...]:
    if component_count <= len(_FORCE_COMPONENT_LABELS):
        return _FORCE_COMPONENT_LABELS[:component_count]
    return tuple(f"F{index}" for index in range(component_count))


def _edge_boundary_colour(edge: Edge) -> str:
    states = {edge.x, edge.y}
    if EEdgeCondition.Traction in states:
        return "tab:orange"
    if EEdgeCondition.Fixed in states:
        return "tab:red"
    return "tab:green"


def _edge_boundary_label(edge_name: str, edge: Edge) -> str:
    return f"{edge_name}\nx={edge.x.name}, y={edge.y.name}"


def _find_traction_edge_name(edge_conditions: EdgeConditions) -> str | None:
    ordered_edges = (
        ("min_x_edge", edge_conditions.min_x_edge),
        ("max_x_edge", edge_conditions.max_x_edge),
        ("min_y_edge", edge_conditions.min_y_edge),
        ("max_y_edge", edge_conditions.max_y_edge),
    )
    for edge_name, edge in ordered_edges:
        if EEdgeCondition.Traction in (edge.x, edge.y):
            return edge_name
    return None


def _resolve_force_arrow_vector(
    force: npt.NDArray[np.float64],
    traction_edge_name: str,
) -> tuple[float, float]:
    """Unit direction of the applied force for the given traction edge."""
    # x-edges carry the x force component (column 0), y-edges the y (column 1).
    is_x_edge = traction_edge_name in ("min_x_edge", "max_x_edge")
    component_index = min(0 if is_x_edge else 1, force.shape[1] - 1)
    component = force[:, component_index]

    nonzero = component[np.isfinite(component) & ~np.isclose(component, 0.0)]
    sign = float(np.sign(nonzero[-1])) if nonzero.size > 0 else 1.0

    return (sign, 0.0) if is_x_edge else (0.0, sign)
