from pathlib import Path

import numpy as np
import numpy.typing as npt

from pyvale.vfm.inputdata import EFeDataSource, InputDataConfig
from pyvale.vfm.interpfedata import (
    interpolate_fe_point_cloud_to_grid,
    load_fe_point_cloud_from_txt_files,
)
from pyvale.vfm.loadmoosedata import (
    load_moose_force,
    load_moose_strain,
    load_moose_timesteps,
)


def preprocess_input_data(config: InputDataConfig) -> tuple[
    npt.NDArray[np.float64],  # x, shape (y, x)
    npt.NDArray[np.float64],  # y, shape (y, x)
    npt.NDArray[np.float64],  # strain, shape (timesteps, components, y, x)
    npt.NDArray[np.float64],  # force, shape (timesteps, components)
    npt.NDArray[np.float64],  # time, shape (timesteps,)
]:
    if config.data_source == EFeDataSource.ANSYS:
        x, y, strain, force, time = _load_ansys_data(config)
    elif config.data_source == EFeDataSource.MOOSE:
        x, y, strain, force, time = _load_moose_data(config)
    else:
        raise ValueError(f"Unsupported data source: {config.data_source}.")

    _validate_input_data(
        x,
        y,
        strain,
        force,
        time
    )

    # Enforce conventions
    # Validate data
    # generate plot
    # save plots
    # save results in a dir

    return (x, y, strain, force, time)


def _load_ansys_data(config: InputDataConfig) -> tuple[
    npt.NDArray[np.float64],  # x, shape (y, x)
    npt.NDArray[np.float64],  # y, shape (y, x)
    npt.NDArray[np.float64],  # strain, shape (timesteps, components, y, x)
    npt.NDArray[np.float64],  # force, shape (timesteps, components)
    npt.NDArray[np.float64],  # time, shape (timesteps,)
]:
    if config.ansys is None:
        raise ValueError(
            "data_source is ANSYS but no ansys config was provided."
        )

    ansys = config.ansys
    fe_dir = Path(ansys.fe_data_dir)

    # Build the FE point cloud from the separate per-component txt files.
    point_cloud = load_fe_point_cloud_from_txt_files(
        x_coordinates_path=fe_dir / ansys.x_coordinates_file,
        y_coordinates_path=fe_dir / ansys.y_coordinates_file,
        time_values_path=fe_dir / ansys.time_values_file,
        component_paths={
            name: fe_dir / file_name
            for name, file_name in ansys.strain_component_files.items()
        },
        element_ids_path=(
            fe_dir / ansys.element_ids_file
            if ansys.element_ids_file is not None
            else None
        ),
    )

    # Interpolate the FE centroid data onto a regular grid.
    interpolated = interpolate_fe_point_cloud_to_grid(
        point_cloud,
        mesh_path=(
            fe_dir / ansys.mesh_file if ansys.mesh_file is not None else None
        ),
        upsample_factor=ansys.upsample_factor,
        target_spacing=ansys.target_spacing,
        source_path=fe_dir,
    )

    # Force and time are not produced by the interpolation, so load them
    # from their configured files. The force file holds a single Fy column;
    # stack a zero Fx column to match the (timesteps, [Fx, Fy]) convention.
    force_fy = config.force.load_from_file()
    force = np.column_stack((np.zeros_like(force_fy), force_fy))
    time = config.time.load_from_file()

    return (
        interpolated.x_grid,
        interpolated.y_grid,
        interpolated.strain,
        force,
        time,
    )


def _load_moose_data(config: InputDataConfig) -> tuple[
    npt.NDArray[np.float64],  # x, shape (y, x)
    npt.NDArray[np.float64],  # y, shape (y, x)
    npt.NDArray[np.float64],  # strain, shape (timesteps, components, y, x)
    npt.NDArray[np.float64],  # force, shape (timesteps, components)
    npt.NDArray[np.float64],  # time, shape (timesteps,)
]:
    if config.moose is None:
        raise ValueError(
            "data_source is MOOSE but no moose config was provided."
        )

    x, y, strain = load_moose_strain(config.moose)
    force = load_moose_force(config.moose)
    time = load_moose_timesteps(config.moose)

    return (x, y, strain, force, time)


def _validate_input_data(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    strain: npt.NDArray[np.float64],
    force: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64]
):
    errors: list[str] = []

    # Check dims
    if x.ndim == 1 and y.ndim == 1:
        x, y = np.meshgrid(x, y)

    if force.ndim != 2:
        errors.append(
            f"Force must be a 2D array. Got a {force.ndim}D array."
        )

    if time.ndim != 1:
        errors.append(
            f"Time must be a 1D array. Got a {time.ndim}D array."
        )

    if strain.ndim != 4:
        errors.append(
            f"Strain must be a 4D array. Got a {strain.ndim}D array."
        )

    # Check shapes
    if x.shape != strain.shape[2:]:
        errors.append(
            f"Coordinate grid shape {x.shape} does not match spatial strain "
            f"components. Got shape {strain.shape[2:]}"
        )

    if force.shape[0] != time.shape[0]:
        errors.append(
            f"Number of rows in force ({force.shape[0]}) does not match the "
            f"number of timesteps ({time.shape[0]})."
        )

    if time.shape[0] != strain.shape[0]:
        errors.append(
            f"Number of timesteps ({time.shape[0]}) does not match the length "
            f"of the strain 0th dimension ({strain.shape[0]})."
        )

    if errors:
        raise ValueError(
            "Invalid input data:\n" + "\n".join(f"  - {e}" for e in errors)
        )
