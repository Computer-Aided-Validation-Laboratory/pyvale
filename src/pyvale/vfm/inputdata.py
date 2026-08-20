from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pyvale.vfm.experimentdata import (
    BoundaryConditions,
    EdgeConditions,
    ExperimentData,
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
    assembled MatchID DIC archive), builds an ``ExperimentData`` object,
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


def _build_experiment_data(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    strain: npt.NDArray[np.float64],
    force: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    thickness: float,
    edge_conditions: EdgeConditions,
    region_of_interest: VfmRegionOfInterest | None = None,
) -> ExperimentData:
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

    return ExperimentData(
        strain,
        specimen_geometry,
        boundary_conditions,
        time,
    )
