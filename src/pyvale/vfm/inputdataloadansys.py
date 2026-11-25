import numpy as np
import numpy.typing as npt

from pyvale.vfm.inputdataconfig import AnsysConfig
from pyvale.vfm.inputdatafeinterp import (
    interpolate_fe_point_cloud_to_grid,
    load_fe_point_cloud_from_txt_files,
)


def load_ansys_data(config: AnsysConfig) -> tuple[
    npt.NDArray[np.float64],  # x, shape (y, x)
    npt.NDArray[np.float64],  # y, shape (y, x)
    npt.NDArray[np.float64],  # strain, shape (timesteps, components, y, x)
    npt.NDArray[np.float64],  # force, shape (timesteps, components)
    npt.NDArray[np.float64],  # time, shape (timesteps,)
]:
    """
    Load Ansys FE results and interpolate them onto a regular grid.

    Builds a point cloud from the per-component txt files described by
    ``config``, interpolates the strain data onto a regular grid, and loads
    the associated reaction force and time data.

    Parameters
    ----------
    config : AnsysConfig
        Paths and interpolation settings for the Ansys data files

    Returns
    -------
    tuple of npt.NDArray[np.float64]
        ``(x, y, strain, force, time)`` where ``x`` and ``y`` are the grid
        coordinates with shape ``(y, x)``, ``strain`` has shape
        ``(timesteps, components, y, x)``, ``force`` has shape
        ``(timesteps, components)``, and ``time`` has shape ``(timesteps,)``
    """

    # Build the FE point cloud from the separate per-component txt files.
    point_cloud = load_fe_point_cloud_from_txt_files(
        config.x_file,
        config.y_file,
        config.time_file,
        component_paths={
            "strain_xx": config.strain_xx_file,
            "strain_yy": config.strain_yy_file,
            "strain_xy": config.strain_xy_file,
        },
        element_ids_path=(
            config.element_ids_file
            if config.element_ids_file is not None
            else None
        ),
    )

    # Interpolate the FE centroid data onto a regular grid.
    interpolated = interpolate_fe_point_cloud_to_grid(
        point_cloud,
        mesh_path=(
            config.mesh_file if config.mesh_file is not None else None
        ),
        upsample_factor=config.upsample_factor,
        target_spacing=config.target_spacing,
    )

    # Force and time are not produced by the interpolation, so load them
    # from their configured files
    force_file_content = np.genfromtxt(
        config.force_file,
        delimiter=",",
        names=True
    )

    force = np.column_stack((
        force_file_content["reaction_fx"],
        force_file_content["reaction_fy"],
    ))

    time = np.genfromtxt(config.time_file)

    return (
        interpolated.x_grid,
        interpolated.y_grid,
        interpolated.strain,
        force,
        time,
    )
