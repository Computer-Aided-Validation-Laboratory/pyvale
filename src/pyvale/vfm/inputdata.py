#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pyvale.dataio import EStrainShear, ExpData
from pyvale.vfm.experimentdata import (
    BoundaryConditions,
    EdgeConditions,
    VfmInputData,
    SpecimenGeometry,
)
from pyvale.vfm.inputdataconfig import AnsysConfig, InputDataConfig, MooseConfig
from pyvale.vfm.inputdataloadansys import load_ansys_data
from pyvale.vfm.inputdataloadmoose import load_moose_data
from pyvale.vfm.inputdatamatchidassembled import (
    MatchIDAssembledConfig,
    load_matchid_assembled_data,
)
from pyvale.vfm.inputdataassembled import AssembledDataConfig, load_assembled_data
from pyvale.vfm.inputdataplots import _create_diagnostic_plots
from pyvale.vfm.roi import VfmRegionOfInterest, convert_mask_to_physical_roi
from pyvale.vfm.validation import run_validation


def process_input_data(
    config: InputDataConfig | MatchIDAssembledConfig | AssembledDataConfig,
    output_root: str | Path = ".",
    *,
    timestamped: bool = True,
) -> Path:
    """
    Load, process, and save experiment data from solver or DIC output.

    Loads the raw field data described by ``config`` (from Ansys, MOOSE, or an
    assembled MatchID DIC archive), builds an ``VfmInputData`` object,
    validates it, writes diagnostic plots, and saves the result to a
    timestamped run directory under ``output_root`` by default. Set
    ``timestamped=False`` to write directly to an explicitly selected output
    directory, for reproducible published datasets.

    Parameters
    ----------
    config : InputDataConfig | MatchIDAssembledConfig
        Source-specific configuration describing the input data files and
        specimen properties.
    output_root : str | Path, optional
        Directory in which the timestamped run directory is created,
        by default ``"."``

    Returns
    -------
    Path
        Path of the newly created ``experiment_data.yaml`` file
    """
    region_of_interest: VfmRegionOfInterest | None = None
    if isinstance(config, AnsysConfig):
        x, y, strain, force, time = load_ansys_data(config)
    elif isinstance(config, MooseConfig):
        x, y, strain, force, time = load_moose_data(config)
    elif isinstance(config, MatchIDAssembledConfig):
        matchid_data = load_matchid_assembled_data(config)
        x = matchid_data.x
        y = matchid_data.y
        strain = matchid_data.strain
        force = matchid_data.force
        time = matchid_data.time
        region_of_interest = matchid_data.region_of_interest
    elif isinstance(config, AssembledDataConfig):
        assembled_data = load_assembled_data(config)
        x = assembled_data.x
        y = assembled_data.y
        strain = assembled_data.strain
        force = assembled_data.force
        time = assembled_data.time
        region_of_interest = assembled_data.region_of_interest
    else:
        raise TypeError(f"Unsupported VFM input-data config: {type(config)!r}")

    experiment_data = _build_experiment_data(
        x,
        y,
        strain,
        force,
        time,
        config.thickness,
        config.edge_conditions,
        region_of_interest,
    )

    run_validation(experiment_data)

    if timestamped:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        run_dir = Path(output_root) / f"vfm-input-data_{timestamp}"
    else:
        run_dir = Path(output_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    _create_diagnostic_plots(
        run_dir / "diagnostic_images",
        x,
        y,
        strain,
        force,
        time,
        config.edge_conditions
    )

    experiment_data.save_to_yaml(run_dir)

    return run_dir / "experiment_data.yaml"


def build_vfm_input_data(
    dic_data: ExpData,
    daq_data: ExpData,
    *,
    thickness: float,
    edge_conditions: EdgeConditions,
    region_of_interest: VfmRegionOfInterest,
    force_x_label: str | None = None,
    force_y_label: str | None = None,
    strain_labels: tuple[str, str, str] = ("strain_xx", "strain_yy", "strain_xy"),
    shear_convention: EStrainShear = EStrainShear.TENSOR,
    coordinate_tolerance: float = 1e-8,
) -> VfmInputData:
    """Assemble solver inputs from DIC and DAQ measurements on matching times.

    Reference coordinates and thickness are in mm. Supply three strain labels
    in xx, yy, xy order and force channel labels for forces in newtons. An
    omitted force direction is set to zero. Coordinates must describe a
    complete, axis-aligned, uniformly spaced planar grid. Unmeasured cells
    may carry NaN strain outside the supplied ROI.
    """
    fields = dic_data.fields
    coords = np.asarray(dic_data.coords, dtype=np.float64)
    if fields.ndim != 3 or coords.shape != (fields.shape[0], 3):
        raise ValueError("DIC data must provide (points, components, times) and XYZ.")
    if not coords.size or not np.all(np.isfinite(coords)):
        raise ValueError("DIC data must provide finite reference coordinates.")
    if not np.isfinite(coordinate_tolerance) or coordinate_tolerance < 0:
        raise ValueError("Coordinate tolerance must be finite and non-negative.")
    if np.ptp(coords[:, 2]) > coordinate_tolerance:
        raise ValueError("DIC reference coordinates must lie in a constant-Z plane.")

    labels = dic_data.component_labels
    if (
        labels is None
        or len(labels) != fields.shape[1]
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("DIC data must provide a unique label for each component.")
    if len(set(strain_labels)) != 3 or not set(strain_labels).issubset(labels):
        raise ValueError(
            f"Supply three distinct DIC strain components {strain_labels}."
        )

    x_axis, columns = _index_uniform_axis(coords[:, 0], coordinate_tolerance)
    y_axis, rows = _index_uniform_axis(coords[:, 1], coordinate_tolerance)
    grid_shape = (y_axis.size, x_axis.size)
    if dic_data.grid_shape is not None and dic_data.grid_shape != grid_shape:
        raise ValueError("DIC grid shape disagrees with its physical coordinates.")

    flat_indices = np.ravel_multi_index((rows, columns), grid_shape)
    if (
        flat_indices.size != x_axis.size * y_axis.size
        or np.unique(flat_indices).size != flat_indices.size
    ):
        raise ValueError("DIC coordinates must cover every grid cell exactly once.")
    point_order = np.argsort(flat_indices)
    component_indices = [labels.index(label) for label in strain_labels]
    strain = np.asarray(
        fields[point_order[:, None], component_indices, :], dtype=np.float64
    ).transpose(2, 1, 0).reshape((fields.shape[2], 3, *grid_shape))
    if shear_convention is EStrainShear.ENGINEERING:
        strain[:, 2] *= 0.5

    time = np.asarray(dic_data.times, dtype=np.float64)
    daq_time = np.asarray(daq_data.times, dtype=np.float64)
    if time.shape != (fields.shape[2],) or not time.size:
        raise ValueError("Supply one DIC timestamp per frame.")
    if not np.array_equal(time, daq_time):
        raise ValueError("DAQ timestamps must match the DIC frame timestamps.")
    if daq_data.fields.ndim != 2 or daq_data.fields.shape[1] != time.size:
        raise ValueError("DAQ data must provide fields with shape (channels, times).")
    if force_x_label is None and force_y_label is None:
        raise ValueError("Supply at least one DAQ force channel label.")

    force = np.zeros((time.size, 2), dtype=np.float64)
    for direction, label in enumerate((force_x_label, force_y_label)):
        if label is None:
            continue
        channel = daq_data.sens_label_to_ind.get(label)
        if channel is None or not 0 <= channel < daq_data.fields.shape[0]:
            raise ValueError(f"Unknown DAQ force channel '{label}'.")
        force[:, direction] = daq_data.fields[channel]

    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError("Specimen thickness must be finite and positive.")
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    vfm_input = _build_experiment_data(
        x_grid, y_grid, strain, force, time.copy(), thickness, edge_conditions,
        region_of_interest,
    )
    run_validation(vfm_input)
    return vfm_input


def _index_uniform_axis(
    coordinates: npt.NDArray[np.float64], tolerance: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
    """Group coordinate repeats and map them onto a uniform increasing axis."""
    order = np.argsort(coordinates)
    sorted_coordinates = coordinates[order]
    group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_coordinates) > tolerance) + 1]
    groups = np.diff(np.r_[group_starts, coordinates.size])
    axis_values = np.add.reduceat(sorted_coordinates, group_starts) / groups
    if axis_values.size < 2:
        raise ValueError("VFM requires at least two distinct coordinates per axis.")
    axis = np.linspace(axis_values[0], axis_values[-1], axis_values.size)
    indices = np.empty(coordinates.size, dtype=np.intp)
    indices[order] = np.repeat(np.arange(axis.size), groups)
    if not np.allclose(coordinates, axis[indices], rtol=0.0, atol=tolerance):
        raise ValueError("DIC coordinates must form a uniformly spaced grid.")
    return axis, indices


def _build_experiment_data(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    strain: npt.NDArray[np.float64],
    force: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    thickness: float,
    edge_conditions: EdgeConditions,
    region_of_interest: VfmRegionOfInterest | None = None,
) -> VfmInputData:

    # If no region of interest is provided, use the finite strain values to define it.
    # Note that this is not the recommended approach for real experiments,
    # but it is useful for testing and debugging. Future versions of this code will
    # seek to provide clearer, more robust guidance on how to define the region of interest
    # for real experiments.
    if region_of_interest is None:
        specimen_mask = np.isfinite(strain[0, 0, :, :])
        region_of_interest = VfmRegionOfInterest.from_definition(
            convert_mask_to_physical_roi(
                specimen_mask,
                x,
                y,
                simplification_pixels=0.0
            )
        )

    element_area = (
        (x[0, 1] - x[0, 0])
        * (y[1, 0] - y[0, 0])
    )

    pixel_area = np.full_like(x, element_area, dtype=np.float64)

    specimen_geometry = SpecimenGeometry(
        x,
        y,
        pixel_area,
        thickness,
        region_of_interest
    )

    boundary_conditions = BoundaryConditions(
        edge_conditions,
        force,
    )

    return VfmInputData(
        strain,
        specimen_geometry,
        boundary_conditions,
        time,
    )
