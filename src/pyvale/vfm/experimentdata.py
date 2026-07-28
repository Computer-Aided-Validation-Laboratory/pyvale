import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt
import yaml

from pyvale.vfm.roi import VfmRegionOfInterest, convert_mask_to_physical_roi


@dataclass(slots=True)
class SpecimenGeometry:
    """
    Physical geometry of the test specimen.

    Stores the spatial coordinates, region-of-interest mask, thickness, and
    per-point physical area of the DIC grid
    """

    x: npt.NDArray[np.float64]
    """
    x-coordinates at each grid point, shape ``(y, x)`` (mm).

    Always positive, increasing left to right (column index)
    """

    y: npt.NDArray[np.float64]
    """
    y-coordinates at each grid point, shape ``(y, x)`` (mm).

    Always positive, increasing top to bottom (row index)
    """

    pixel_area: npt.NDArray[np.float64]
    """Area per grid point, shape ``(y, x)`` (mm²)"""

    thickness: float
    """Out-of-plane thickness of the specimen (mm)"""

    # TODO: docstring
    region_of_interest: VfmRegionOfInterest


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
    """
    Boundary conditions on the four edges of the specimen.

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
    """
    Measured force history, shape ``(timesteps, 2)`` with columns
    ``[Fx, Fy]`` (x-direction, y-direction)
    """


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

    @classmethod
    def load_from_file(cls, experiment_data_file: str | Path) -> Self:
        """
        Build an ``ExperimentData`` from a saved yaml file.
        """

        experiment_data_file = Path(experiment_data_file)
        base_dir = experiment_data_file.parent

        experiment_data_file_content = yaml.safe_load(
            experiment_data_file.read_text(encoding="utf-8")
        )

        x_path = base_dir / experiment_data_file_content["x"]
        y_path = base_dir / experiment_data_file_content["y"]
        strain_path = base_dir / experiment_data_file_content["strain"]
        force_path = base_dir / experiment_data_file_content["force"]
        timesteps_path = base_dir / experiment_data_file_content["time"]

        roi_path = (
            base_dir / experiment_data_file_content["region_of_interest"]
        )

        thickness = experiment_data_file_content["thickness"]

        edge_conditions = experiment_data_file_content["edge_conditions"]

        x = np.load(x_path)
        y = np.load(y_path)
        strain = np.load(strain_path)
        force = np.load(force_path)
        timesteps = np.load(timesteps_path)

        if roi_path.exists():
            roi = VfmRegionOfInterest.from_yaml(roi_path)
        else:
            print(
                "Note: No region_of_interest.yaml found, generating a region "
                "of interest from nan values in strain data"
            )
            specimen_mask = np.isfinite(strain[0, 0, :, :])

            roi = VfmRegionOfInterest.from_definition(
                convert_mask_to_physical_roi(
                    specimen_mask,
                    x,
                    y,
                    simplification_pixels=0.0
                )
            )


        edge_conditions = _unpack_edge_conditions(edge_conditions)

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
            roi
        )

        boundary_conditions = BoundaryConditions(
            edge_conditions,
            force
        )

        return cls(
            strain,
            specimen_geometry,
            boundary_conditions,
            timesteps,
        )

    def save_to_yaml(self, output_dir: str | Path):
        """
        Save this ``ExperimentData`` into ``output_dir`` with an
        ``experiment_data.yaml`` and sibling npy files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        experiment_data_file_content = {
            "x": "x.npy",
            "y": "y.npy",
            "strain": "strain.npy",
            "force": "force.npy",
            "time": "time.npy",
            "region_of_interest": "region_of_interest.yaml",
            "thickness": self.specimen_geometry.thickness,
            "edge_conditions": _pack_edge_conditions(
                self.boundary_conditions.edge_conditions
            )
        }

        np.save(output_dir / "x.npy", self.specimen_geometry.x)
        np.save(output_dir / "y.npy", self.specimen_geometry.y)
        np.save(output_dir / "strain.npy", self.strain)
        np.save(output_dir / "force.npy", self.boundary_conditions.force)
        np.save(output_dir / "time.npy", self.timesteps)

        self.specimen_geometry.region_of_interest.save_yaml(
            output_dir / "region_of_interest.yaml"
        )

        experiment_data_file = output_dir / "experiment_data.yaml"

        experiment_data_file.write_text(
            yaml.safe_dump(experiment_data_file_content, sort_keys=False),
            encoding="utf-8"
        )


def _pack_edge_conditions(
    edge_conditions: EdgeConditions,
) -> dict[str, dict[str, str]]:
    return {
        name: {"x": edge.x.name, "y": edge.y.name}
        for name, edge in (
            ("min_x_edge", edge_conditions.min_x_edge),
            ("max_x_edge", edge_conditions.max_x_edge),
            ("min_y_edge", edge_conditions.min_y_edge),
            ("max_y_edge", edge_conditions.max_y_edge),
        )
    }


def _unpack_edge_conditions(
    data: dict[str, dict[str, str]]
) -> EdgeConditions:
    return EdgeConditions(
        min_x_edge=Edge(
            EEdgeCondition[data["min_x_edge"]["x"]],
            EEdgeCondition[data["min_x_edge"]["y"]]
        ),
        max_x_edge=Edge(
            EEdgeCondition[data["max_x_edge"]["x"]],
            EEdgeCondition[data["max_x_edge"]["y"]]
        ),
        min_y_edge=Edge(
            EEdgeCondition[data["min_y_edge"]["x"]],
            EEdgeCondition[data["min_y_edge"]["y"]]
        ),
        max_y_edge=Edge(
            EEdgeCondition[data["max_y_edge"]["x"]],
            EEdgeCondition[data["max_y_edge"]["y"]]
        )
    )
