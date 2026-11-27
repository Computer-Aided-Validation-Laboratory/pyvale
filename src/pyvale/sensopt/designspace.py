"""Position domains and geometry preprocessing for sensor placement."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import numpy.typing as npt

from pyvale.dataio.simdata import EMeshType, SimData
from pyvale.sensorsim.enums import EDim
from pyvale.sensorsim.fieldconverter import simdata_to_pyvista_vis
from pyvale.sensorsim.sensordata import SensorData


class EDesignAxis(Enum):
    """Representation of an optimisation domain."""

    DISC = auto()


class IConstraintInput(ABC):
    """Feasibility condition on the proposed layout before sensor simulation."""

    @abstractmethod
    def eval(self, sensor_data: Mapping[str, SensorData]) -> npt.NDArray[np.float64]:
        """Return constraint residuals. Each residual must be nonpositive."""


@dataclass(slots=True, kw_only=True)
class DesignSpace:
    """Candidate positions and input constraints for the named sensor arrays.

    Each discrete position domain is shaped (site, xyz) in the field's coordinate
    system and units. The row order breaks ties during search.
    """

    positions: Mapping[str, npt.NDArray[np.float64]]
    position_axes: Mapping[str, EDesignAxis]
    input_constraints: tuple[IConstraintInput, ...] = ()


def filter_sensor_sites_in_mesh(
    sites: npt.NDArray[np.float64], sim_data: SimData
) -> npt.NDArray[np.float64]:
    """Keep candidate sites contained in the simulation mesh.

    Parameters
    ----------
    sites : numpy.ndarray
        Coordinates shaped (site, xyz), in the same frame and units as the mesh.
    sim_data : SimData
        Simulation with a surface mesh or a volume mesh.

    Returns
    -------
    numpy.ndarray
        Copy of the contained sites in their original order, shaped (site, xyz).

    Raises
    ------
    ValueError
        If sites are malformed or the simulation has no mesh.
    """
    if sites.ndim != 2 or sites.shape[1] != 3 or not np.all(np.isfinite(sites)):
        raise ValueError("Candidate sites must be finite and have shape (site, 3).")
    if sim_data.coords is None or not sim_data.connect:
        raise ValueError("Filtering sensor sites requires a mesh with connectivity.")

    match sim_data.mesh_type:
        case EMeshType.SURF:
            dimensions = EDim.TWOD
        case EMeshType.VOL:
            dimensions = EDim.THREED
        case _:
            raise ValueError("Sensor sites require a surface or volume mesh.")
    if sites.shape[0] == 0:
        return sites.copy()

    mesh = simdata_to_pyvista_vis(sim_data, dimensions)
    in_mesh = np.asarray(mesh.find_containing_cell(sites)) >= 0
    return sites[in_mesh]
