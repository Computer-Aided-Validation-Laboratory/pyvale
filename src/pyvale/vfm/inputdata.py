from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pyvale.vfm.ansysloaddata import load_ansys_data
from pyvale.vfm.experimentdata import (
    BoundaryConditions,
    EdgeConditions,
    ExperimentData,
    SpecimenGeometry,
)
from pyvale.vfm.inputdataconfig import AnsysConfig, InputDataConfig, MooseConfig
from pyvale.vfm.inputdataplots import _create_diagnostic_plots
from pyvale.vfm.mooseloaddata import load_moose_data
from pyvale.vfm.roi import VfmRegionOfInterest, convert_mask_to_physical_roi
from pyvale.vfm.validation import validate_experiment_data


# TODO: docs
# returns path of newly created experiment data file
def process_input_data(
    config: InputDataConfig,
    output_root: str | Path = "."
) -> Path:
    if isinstance(config, AnsysConfig):
        x, y, strain, force, time = load_ansys_data(config)
    elif isinstance(config, MooseConfig):
        x, y, strain, force, time = load_moose_data(config)

    experiment_data = _build_experiment_data(
        x,
        y,
        strain,
        force,
        time,
        config.thickness,
        config.edge_conditions
    )

    validate_experiment_data(experiment_data)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    run_dir = Path(output_root) / f"vfm-input-data_{timestamp}"
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


# TODO: this currently only supports getting roi from nans in strain data
def _build_experiment_data(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    strain: npt.NDArray[np.float64],
    force: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    thickness: float,
    edge_conditions: EdgeConditions
) -> ExperimentData:
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
