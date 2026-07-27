from dataclasses import dataclass
from pathlib import Path

from pyvale.vfm.experimentdata import EdgeConditions


@dataclass(slots=True)
class AnsysConfig:
    """
    Inputs for interpolating ANSYS FE centroid data onto a regular grid.
    """
    x_file: Path
    y_file: Path

    strain_xx_file: Path
    strain_yy_file: Path
    strain_xy_file: Path

    force_file: Path
    time_file: Path

    thickness: float
    edge_conditions: EdgeConditions

    element_ids_file: Path | None = None
    mesh_file: Path | None = None

    upsample_factor: float = 2.0
    target_spacing: float | None = None


@dataclass(slots=True)
class MooseConfig:
    """
    Inputs for loading MOOSE exodus data and interpolating onto a grid.
    """
    exodus_file_path: str

    height: float
    width: float
    thickness: float

    grid_divs: int

    edge_conditions: EdgeConditions

    strain_component_keys: tuple[str, str, str] = (
        "strain_xx",
        "strain_yy",
        "strain_xy",
    )

    force_key: str = "react_y_top"


InputDataConfig = AnsysConfig | MooseConfig
