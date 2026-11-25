from dataclasses import dataclass
from pathlib import Path

from pyvale.vfm.experimentdata import EdgeConditions


@dataclass(slots=True)
class AnsysConfig:
    """
    Inputs for interpolating ANSYS FE centroid data onto a regular grid.
    """

    x_file: Path
    """Path to the txt file of FE centroid x-coordinates"""

    y_file: Path
    """Path to the txt file of FE centroid y-coordinates"""

    strain_xx_file: Path
    """Path to the txt file of xx-component strain values"""

    strain_yy_file: Path
    """Path to the txt file of yy-component strain values"""

    strain_xy_file: Path
    """Path to the txt file of xy-component (shear) strain values"""

    force_file: Path
    """
    Path to the csv file of reaction forces ``reaction_fx`` and ``reaction_fy``
    """

    time_file: Path
    """Path to the txt file of time values, one per timestep"""

    thickness: float
    """Out-of-plane thickness of the specimen (mm)"""

    edge_conditions: EdgeConditions
    """Boundary conditions applied to the four edges of the specimen"""

    element_ids_file: Path | None = None
    """Optional path to the txt file of FE element ids for the centroids"""

    mesh_file: Path | None = None
    """Optional path to the FE mesh file used to bound the interpolation grid"""

    upsample_factor: float = 2.0
    """
    Factor by which the grid is refined relative to the FE centroid spacing.
    Ignored when ``target_spacing`` is set
    """

    target_spacing: float | None = None
    """Explicit grid spacing (mm); overrides ``upsample_factor`` when set"""


@dataclass(slots=True)
class MooseConfig:
    """
    Inputs for loading MOOSE exodus data and interpolating onto a grid.
    """

    exodus_file_path: str
    """Path to the MOOSE exodus output file"""

    height: float
    """Height of the specimen (mm)"""

    width: float
    """Width of the specimen (mm)"""

    thickness: float
    """Out-of-plane thickness of the specimen (mm)"""

    grid_divs: int
    """Number of grid divisions along each axis of the interpolation grid"""

    edge_conditions: EdgeConditions
    """Boundary conditions applied to the four edges of the specimen"""

    strain_component_keys: tuple[str, str, str] = (
        "strain_xx",
        "strain_yy",
        "strain_xy",
    )
    """
    Exodus field keys for the strain components, ordered as
    ``(xx, yy, xy)``
    """

    force_key: str = "react_y_top"
    """Exodus global variable key for the reaction force"""


InputDataConfig = AnsysConfig | MooseConfig
"""
Solver-specific input configuration, either an ``AnsysConfig`` or a
``MooseConfig``. Determines which loader ``process_input_data`` uses to
produce the ``ExperimentData``.
"""
