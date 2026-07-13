import enum
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class SpecimenGeometry:
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    region_of_interest: npt.NDArray[np.bool_]
    thickness: float
    pixel_area: npt.NDArray[np.float64]


class EEdgeCondition(enum.Enum):
    Free = enum.auto()
    Fixed = enum.auto()
    Traction = enum.auto()


@dataclass(slots=True)
class Edge:
    x: EEdgeCondition
    y: EEdgeCondition


@dataclass(slots=True)
class EdgeConditions:
    min_x_edge: Edge
    max_x_edge: Edge
    min_y_edge: Edge
    max_y_edge: Edge


@dataclass(slots=True)
class BoundaryConditions:
    edge_conditions: EdgeConditions
    force: npt.NDArray[np.float64]


def _calculate_timestep_deltas(
    timesteps: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    delta_timesteps = np.zeros_like(timesteps)

    delta_timesteps[0] = timesteps[0]
    delta_timesteps[1:] = np.diff(timesteps)

    return delta_timesteps


@dataclass(slots=True)
class ExperimentData:
    strain: npt.NDArray[np.float64]
    specimen_geometry: SpecimenGeometry
    boundary_conditions: BoundaryConditions
    timesteps: npt.NDArray[np.float64]
    delta_timesteps: npt.NDArray[np.float64]

    def __init__(
        self,
        strain: npt.NDArray[np.float64],
        specimen_geometry: SpecimenGeometry,
        boundary_conditions: BoundaryConditions,
        timesteps: npt.NDArray[np.float64],
    ) -> None:
        self.strain = strain
        self.specimen_geometry = specimen_geometry
        self.boundary_conditions = boundary_conditions
        self.timesteps = timesteps
        self.delta_timesteps = _calculate_timestep_deltas(self.timesteps)
