"""Concrete objectives against the proposed OptSens interface."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree

# pyvale imports
import pyvale.sensopt as opt
import pyvale.sensorsim as sens


@dataclass(slots=True)
class ObjNodalReconRMSE(opt.IObjFunc):
    target_positions: npt.NDArray[np.float64]
    target_values: Mapping[str, npt.NDArray[np.float64]]
    sensor_keys: tuple[str, ...]
    component_index: int

    def get_required_outputs(self) -> frozenset[sens.EExpSimKeys]:
        return frozenset({sens.EExpSimKeys.MEASUREMENT})

    def eval(
        self,
        *,
        sensor_data: Mapping[str, sens.SensorData],
        experiment_data: opt.ExperimentData,
        scenario_keys: tuple[str, ...],
    ) -> npt.NDArray[np.float64]:
        active_keys: tuple[str, ...] = tuple(
            key for key in self.sensor_keys if key in sensor_data
        )
        if not active_keys:
            raise ValueError("Nodal reconstruction requires an active sensor array.")

        positions: list[npt.NDArray[np.float64]] = []
        for key in active_keys:
            sensor_positions: npt.NDArray[np.float64] | None = sensor_data[
                key
            ].positions
            if sensor_positions is None:
                raise ValueError(f"Sensor array {key!r} requires positions.")
            positions.append(sensor_positions)

        # The same geometry and candidate layout are used in these examples.
        tree: KDTree = KDTree(np.concatenate(positions, axis=0), leafsize=16)
        _, nearest_indices = tree.query(self.target_positions)
        nearest_sensors: npt.NDArray[np.intp] = np.asarray(nearest_indices)
        losses: list[npt.NDArray[np.float64]] = []
        for scenario_key in scenario_keys:
            measurements: npt.NDArray[np.float64] = np.concatenate(
                [
                    experiment_data[scenario_key, key, sens.EExpSimKeys.MEASUREMENT]
                    for key in active_keys
                ],
                axis=1,
            )
            # The selected component uses a common time grid and equal nodal weights.
            predicted: npt.NDArray[np.float64] = measurements[
                :, nearest_sensors, self.component_index, :
            ]
            error: npt.NDArray[np.float64] = (
                predicted - self.target_values[scenario_key]
            )
            losses.append(np.sqrt(np.mean(np.square(error), axis=(1, 2))))

        return np.stack(losses, axis=0)
