import enum
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class SpecimenGeometry:
    """Physical geometry of the test specimen.

    Stores the spatial coordinates, region-of-interest mask, thickness, and
    per-point physical area of the DIC grid
    """

    x: npt.NDArray[np.float64]
    """x-coordinates at each grid point, shape ``(y, x)`` (mm).
    Always positive, increasing left to right (column index)"""

    y: npt.NDArray[np.float64]
    """y-coordinates at each grid point, shape ``(y, x)`` (mm).
    Always positive, increasing top to bottom (row index)"""

    region_of_interest: npt.NDArray[np.bool_]
    """Boolean mask of valid analysis points, shape ``(y, x)``"""

    thickness: float
    """Out-of-plane thickness of the specimen (mm)"""

    pixel_area: npt.NDArray[np.float64]
    """Area per grid point, shape ``(y, x)`` (mm²)"""


class EEdgeCondition(enum.Enum):
    """Mechanical condition applied to an edge of the specimen"""

    Free = enum.auto()
    """Unconstrained edge (stress-free)"""

    Fixed = enum.auto()
    """Fully constrained edge (zero displacement)"""

    Traction = enum.auto()
    """Edge with a known applied traction (force) applied"""


@dataclass(slots=True)
class Edge:
    """Boundary condition for the two orthogonal directions on a single edge"""

    x: EEdgeCondition
    """Condition in the global x-direction"""

    y: EEdgeCondition
    """Condition in the global y-direction"""


@dataclass(slots=True)
class EdgeConditions:
    """Boundary conditions on the four edges of the specimen.

    Edges are identified by the minimum/maximum coordinate value along each
    axis
    """

    min_x_edge: Edge
    """Condition along the minimum x edge"""

    max_x_edge: Edge
    """Condition along the maximum x edge"""

    min_y_edge: Edge
    """Condition along the minimum y edge"""

    max_y_edge: Edge
    """Condition along the maximum y edge"""


@dataclass(slots=True)
class BoundaryConditions:
    """Combined kinematic and kinetic boundary conditions for the experiment"""

    edge_conditions: EdgeConditions
    """Kinematic constraints on all four edges of the specimen"""

    force: npt.NDArray[np.float64]
    """Measured force history, shape ``(timesteps, 2)`` with columns
    ``[Fx, Fy]`` (x-direction, y-direction)"""


def _calculate_timestep_deltas(
    timesteps: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    delta_timesteps = np.zeros_like(timesteps)

    delta_timesteps[0] = timesteps[0]
    delta_timesteps[1:] = np.diff(timesteps)

    return delta_timesteps


@dataclass(slots=True)
class ExperimentData:
    """
    Input data from a DIC experiment.

    Stores the full-field strain history, specimen geometry, boundary
    conditions, and temporal data needed to perform VFM identification.

    Shape conventions
    -----------------
    strain                 (timesteps, components, y, x)
    specimen_geometry:
        x                  (y, x)
        y                  (y, x)
        pixel_area         (y, x)
        region_of_interest (y, x)
    boundary_conditions:
        force              (timesteps, 2)  ``[Fx, Fy]``
    timesteps              (timesteps,)

    Coordinate system
    -----------------
    ``x`` increases left to right (column index)
    ``y`` increases top to bottom (row index)
    All coordinates are always positive, and start at 0.0

    Notes
    -----
    ``delta_timesteps`` is computed automatically from ``timesteps`` on init
    and is not user-supplied
    """

    strain: npt.NDArray[np.float64]
    """
    Full-field strain history, shape ``(timesteps, components, y, x)``
    where ``x`` increases left to right and ``y`` increases top to bottom.
    Components are ordered as ``[xx, yy, xy]`` (normal x, normal y, shear xy)
    """

    specimen_geometry: SpecimenGeometry
    """Geometry of the specimen"""

    boundary_conditions: BoundaryConditions
    """Kinematic and kinetic boundary conditions applied during the test"""

    timesteps: npt.NDArray[np.float64]
    """Time value at each frame / load step, shape ``(timesteps,)``"""

    delta_timesteps: npt.NDArray[np.float64]
    """
    Time increment between consecutive frames (computed automatically),
    shape ``(timesteps,)``
    """

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
