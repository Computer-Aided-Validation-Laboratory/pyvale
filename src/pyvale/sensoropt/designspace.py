# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Position domains and geometry preprocessing for sensor placement."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import combinations_with_replacement
from numbers import Integral

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist, pdist

from pyvale.dataio.simdata import EMeshType, SimData
from pyvale.sensorsim.enums import EDim, EExpSimKeys
from pyvale.sensorsim.fieldconverter import simdata_to_pyvista_vis
from pyvale.sensorsim.sensordata import SensorData


class EDesignAxis(Enum):
    """Representation of an optimisation domain."""

    DISC = auto()


class IConstrainInp(ABC):
    """Feasibility condition on the proposed layout before sensor simulation."""

    @abstractmethod
    def eval(self, sensor_data: Mapping[str, SensorData]) -> npt.NDArray[np.float64]:
        """Return constraint residuals. Each residual must be nonpositive."""


class IConstrainOut(ABC):
    """Feasibility condition on a layout's simulated outputs."""

    @abstractmethod
    def get_required_outputs(self) -> frozenset[EExpSimKeys]:
        """Declare which experiment outputs this constraint reads."""

    @abstractmethod
    def eval(
        self,
        *,
        sensor_data: Mapping[str, SensorData],
        experiment_data: Mapping[tuple[str, str, EExpSimKeys], npt.NDArray[np.float64]],
        scenario_keys: tuple[str, ...],
    ) -> npt.NDArray[np.float64]:
        """Return finite residuals. Each residual must be nonpositive."""


@dataclass(slots=True, kw_only=True)
class ConstrainSensSep(IConstrainInp):
    """Reject co-location and enforce an optional minimum distance.

    Positions across all arrays must share a coordinate frame and units.
    """

    min_distance: float = 0.0
    pair_distances: Mapping[tuple[str, str], float] = field(default_factory=dict)

    def eval(self, sensor_data: Mapping[str, SensorData]) -> npt.NDArray[np.float64]:
        if not np.isfinite(self.min_distance) or self.min_distance < 0:
            raise ValueError("min_distance must be finite and nonnegative.")
        positions = {
            key: data.positions
            for key, data in sensor_data.items()
            if data.positions is not None
        }
        overrides: dict[tuple[str, str], float] = {}
        for (first, second), distance in self.pair_distances.items():
            if not np.isfinite(distance) or distance < 0:
                raise ValueError("Pair distances must be finite and nonnegative.")
            pair = (first, second) if first <= second else (second, first)
            if pair in overrides and overrides[pair] != distance:
                raise ValueError(f"Conflicting separation distances for {pair!r}.")
            overrides[pair] = distance

        if not positions:
            return np.empty(0, dtype=np.float64)
        if not overrides:
            distances = np.asarray(
                pdist(np.concatenate(list(positions.values()))), dtype=np.float64
            )
            if self.min_distance == 0:
                return (distances == 0).astype(np.float64)
            return self.min_distance - distances

        residuals: list[npt.NDArray[np.float64]] = []
        for first, second in combinations_with_replacement(sorted(positions), 2):
            if first == second:
                pair_distances = pdist(positions[first])
            else:
                pair_distances = cdist(positions[first], positions[second]).ravel()
            distances = np.asarray(pair_distances, dtype=np.float64)
            minimum = overrides.get((first, second), self.min_distance)
            if minimum == 0:
                residuals.append((distances == 0).astype(np.float64))
            else:
                residuals.append(minimum - distances)
        return np.concatenate(residuals)


def _calculate_sensor_cost(
    sensor_data: Mapping[str, SensorData], unit_costs: Mapping[str, float]
) -> float:
    cost = 0.0
    for key, unit_cost in unit_costs.items():
        if not np.isfinite(unit_cost) or unit_cost < 0:
            raise ValueError("Sensor unit costs must be finite and nonnegative.")
        if key in sensor_data:
            positions = sensor_data[key].positions
            if positions is None:
                raise ValueError(f"Sensor array {key!r} requires positions.")
            cost += len(positions) * unit_cost
    return cost


@dataclass(slots=True, kw_only=True)
class ConstrainSensCost(IConstrainInp):
    """Enforce a budget for the configured per-array unit costs."""

    unit_costs: Mapping[str, float]
    budget: float

    def eval(self, sensor_data: Mapping[str, SensorData]) -> npt.NDArray[np.float64]:
        if not np.isfinite(self.budget) or self.budget < 0:
            raise ValueError("The sensor budget must be finite and nonnegative.")
        cost = _calculate_sensor_cost(sensor_data, self.unit_costs)
        return np.array([cost - self.budget], dtype=np.float64)


@dataclass(slots=True, kw_only=True)
class ConstrainChannCount(IConstrainInp):
    """Limit DAQ channels using explicit per-sensor hardware channel counts.

    Unselected arrays consume zero channels. Each selected array must have a
    configured count. Use zero channels for hardware that consumes none.
    """

    channels_per_sensor: Mapping[str, int]
    max_channels: int

    def eval(self, sensor_data: Mapping[str, SensorData]) -> npt.NDArray[np.float64]:
        if not isinstance(self.max_channels, Integral) or (
            isinstance(self.max_channels, bool) or self.max_channels < 0
        ):
            raise ValueError("max_channels must be a nonnegative integer.")
        for key, count in self.channels_per_sensor.items():
            if not isinstance(count, Integral) or isinstance(count, bool) or count < 0:
                raise ValueError(
                    f"Channel count for {key!r} must be a nonnegative integer."
                )

        used_channels = 0
        for key, data in sensor_data.items():
            positions = data.positions
            if positions is None:
                raise ValueError(f"Sensor array {key!r} requires positions.")
            if len(positions) == 0:
                continue
            if key not in self.channels_per_sensor:
                raise ValueError(f"Specify channels_per_sensor for array {key!r}.")
            used_channels += len(positions) * int(self.channels_per_sensor[key])
        return np.array([used_channels - int(self.max_channels)], dtype=np.float64)


@dataclass(slots=True, kw_only=True)
class DesignSpace:
    """Candidate positions and input constraints for the named sensor arrays.

    Each discrete position domain is shaped (site, xyz) in the field's coordinate
    system and units. The row order breaks ties during search.
    Each entry in sample_times supplies complete candidate schedules for one
    array. A schedule is a strictly increasing 1D array shared by that array.
    Omit an array to use its configured times.
    """

    positions: Mapping[str, npt.NDArray[np.float64]] = field(default_factory=dict)
    position_axes: Mapping[str, EDesignAxis] = field(default_factory=dict)
    input_constraints: tuple[IConstrainInp, ...] = ()
    output_constraints: tuple[IConstrainOut, ...] = ()
    sample_times: Mapping[str, tuple[npt.NDArray[np.float64], ...]] = field(
        default_factory=dict
    )


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
