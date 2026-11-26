"""Write two CSV exports of the packaged plate-with-hole simulation."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv

from pyvale import data, mooseherder
from pyvale.sensorsim.enums import EDim
from pyvale.sensorsim.fieldconverter import simdata_to_pyvista_interp


@dataclass(slots=True)
class SyntheticDicFiles:
    column_frames: dict[int, Path]
    matrix_frames: dict[int, dict[str, Path]]
    coordinate_files: tuple[Path, Path]
    daq_file: Path
    component_labels: tuple[str, ...]


def write_synthetic_dic_experiment(output_path: Path) -> SyntheticDicFiles:
    simulation = mooseherder.ExodusLoader(data.mechanical_2d_path()).load_all_sim_data()
    if simulation.coords is None or simulation.time is None:
        raise ValueError("The plate simulation must provide coordinates and time.")
    if simulation.glob_vars is None:
        raise ValueError("The plate simulation must provide a top-edge reaction.")

    component_labels = ("disp_x", "disp_y", "strain_xx", "strain_yy", "strain_xy")
    mesh = simdata_to_pyvista_interp(simulation, component_labels, EDim.TWOD)
    min_x, max_x, min_y, max_y, *_ = mesh.bounds
    x_grid, y_grid = np.meshgrid(
        np.linspace(min_x, max_x, 21), np.linspace(min_y, max_y, 31)
    )
    coords = np.column_stack(
        (x_grid.ravel(), y_grid.ravel(), np.full(x_grid.size, simulation.coords[0, 2]))
    )
    sampled = pv.PolyData(coords).sample(mesh)
    valid_points = np.asarray(sampled["vtkValidPointMask"], dtype=bool)
    frame_indices = np.linspace(0, simulation.time.size - 1, 5, dtype=int)
    fields = np.stack(
        [np.asarray(sampled[label])[:, frame_indices] for label in component_labels],
        axis=1,
    )
    fields[~valid_points] = np.nan

    output_path.mkdir(parents=True, exist_ok=True)
    coordinate_files = (output_path / "x.csv", output_path / "y.csv")
    for path, grid in zip(coordinate_files, (x_grid, y_grid), strict=True):
        np.savetxt(path, grid, delimiter=",")

    column_frames: dict[int, Path] = {}
    matrix_frames: dict[int, dict[str, Path]] = {}
    point_ids = np.arange(coords.shape[0])
    rng = np.random.default_rng(17)
    for frame_id in range(frame_indices.size):
        frame_values = fields[:, :, frame_id]
        column_path = output_path / f"frame_{frame_id:03}.csv"
        table = np.column_stack((point_ids, coords, frame_values))
        if frame_id == 1:
            # Point IDs allow the loader to recover reference order.
            table = table[rng.permutation(table.shape[0])]
        np.savetxt(
            column_path,
            table,
            delimiter=",",
            header=",".join(("point_id", "x", "y", "z", *component_labels)),
            comments="",
            fmt=["%d"] + ["%.17g"] * (table.shape[1] - 1),
        )
        column_frames[frame_id] = column_path
        matrix_frames[frame_id] = {}
        for component_index, label in enumerate(component_labels):
            path = output_path / f"frame_{frame_id:03}_{label}.csv"
            np.savetxt(
                path, frame_values[:, component_index].reshape(x_grid.shape),
                delimiter=",",
            )
            matrix_frames[frame_id][label] = path

    daq_file = output_path / "daq.csv"
    daq_values = np.column_stack(
        (
            np.arange(frame_indices.size),
            simulation.time[frame_indices],
            simulation.glob_vars["react_y_top"][frame_indices],
        )
    )
    np.savetxt(
        daq_file, daq_values, delimiter=",", header="frame_id,time,force_y",
        comments="", fmt=["%d", "%.17g", "%.17g"],
    )

    return SyntheticDicFiles(column_frames, matrix_frames, coordinate_files,
                             daq_file, component_labels)
