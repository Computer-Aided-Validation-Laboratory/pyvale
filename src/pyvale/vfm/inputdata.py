from dataclasses import dataclass

from pyvale.vfm.inputdatafiles import InputDataFile, NpyFile
from pyvale.vfm.experimentdata import EdgeConditions
import enum

import numpy.typing as npt
import numpy as np

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
    strain: StrainConfig
    force: ForceConfig
    time: TimeConfig
    region_of_interest: ROIConfig
    thickness: float
    # computed from x and y?
    # pixel_area: float = None
    edge_conditions: EdgeConditions


