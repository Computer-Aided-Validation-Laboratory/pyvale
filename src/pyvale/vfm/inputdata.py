from dataclasses import dataclass, field

from pyvale.vfm.inputdatafiles import InputDataFile, NpyFile
from pyvale.vfm.experimentdata import EdgeConditions
import enum

import numpy.typing as npt
import numpy as np


class EFeDataSource(enum.Enum):
    ANSYS = enum.auto()
    MOOSE = enum.auto()


def _default_strain_component_files() -> dict[str, str]:
    # Maps component name -> txt filename, in [xx, yy, xy] order.
    return {
        "strain_xx": "eps_xx.txt",
        "strain_yy": "eps_yy.txt",
        "strain_xy": "eps_xy.txt",
    }


@dataclass(slots=True)
class AnsysConfig:
    """Inputs for interpolating ANSYS FE centroid data onto a regular grid.

    Strain components are read from separate per-component txt files (e.g.
    eps_xx.txt) since a combined element_data.csv may not exist. All file
    names are resolved relative to ``fe_data_dir``.
    """
    fe_data_dir: str
    strain_component_files: dict[str, str] = field(
        default_factory=_default_strain_component_files
    )
    x_coordinates_file: str = "x_coordinates.txt"
    y_coordinates_file: str = "y_coordinates.txt"
    time_values_file: str = "time_values.txt"
    element_ids_file: str | None = "element_ids.txt"
    mesh_file: str | None = None
    upsample_factor: float = 2.0
    target_spacing: float | None = None


@dataclass(slots=True)
class MooseConfig:
    """Inputs for loading MOOSE exodus data and interpolating onto a grid."""
    exodus_file_path: str
    grid_divs: int
    plate_height: float
    plate_width: float
    strain_component_keys: tuple[str, str, str] = (
        "strain_xx",
        "strain_yy",
        "strain_xy",
    )
    force_key: str = "react_y_top"


@dataclass(slots=True)
class CoordConfig:
    file: InputDataFile

    def load_from_file(self) -> npt.NDArray[np.float64]:
        return self.file.load()


@dataclass(slots=True)
class StrainConfig:
    # if its an npy file with all components
    # we need the index of each field, and the order
    # of components
    #
    # if its fe data, there could be like 4 data points per component
    #
    # if its in a csv file there could be a row per component
    file: InputDataFile

    timestep_dim_index: int
    components_dim_index: int
    y_dim_index: int
    x_dim_index: int

    xx_component_index: int
    yy_component_index: int
    xy_component_index: int

    def __post_init__(self) -> None:
        allowed_file_types = {NpyFile}

        if type(self.file) not in allowed_file_types:
            raise TypeError(
                f"Strain file must be one of {allowed_file_types}. "
                f"Got {type(self.file).__name__}."
            )

    def load_from_file(self) -> npt.NDArray[np.float64]:
        data = self.file.load()

        # Reorder the axes into (timesteps, components, y, x)
        data = np.transpose(
            data,
            (
                self.timestep_dim_index,
                self.components_dim_index,
                self.y_dim_index,
                self.x_dim_index,
            ),
        )

        # Reorder the components axis into [xx, yy, xy]
        data = data[
            :,
            [
                self.xx_component_index,
                self.yy_component_index,
                self.xy_component_index,
            ],
            :,
            :,
        ]

        return data


class EForceUnits(enum.Enum):
    N = enum.auto()
    KN = enum.auto()


@dataclass(slots=True)
class ForceConfig:
    file: InputDataFile
    units: EForceUnits

    should_flip_sign: bool = False

    def load_from_file(self) -> npt.NDArray[np.float64]:
        return self.file.load()


@dataclass(slots=True)
class TimeConfig:
    file: InputDataFile

    should_offset_start_time_to_zero: bool = False

    def load_from_file(self) -> npt.NDArray[np.float64]:
        return self.file.load()


@dataclass(slots=True)
class ROIConfig:
    file: InputDataFile

    def load_from_file(self) -> dict:
        return self.file.load()


@dataclass(slots=True)
class InputDataConfig:
    x: CoordConfig
    y: CoordConfig
    force: ForceConfig
    time: TimeConfig
    thickness: float
    # computed from x and y?
    # pixel_area: float = None
    edge_conditions: EdgeConditions

    # Which FE solver produced the data. ANSYS data is interpolated onto a
    # grid from element centroids; MOOSE data is loaded from an exodus file.
    data_source: EFeDataSource
    ansys: AnsysConfig | None = None
    moose: MooseConfig | None = None


