# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Reconstruct fields from observations at scattered sensor sites."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import KDTree


class IFieldRecon(ABC):
    """Reconstruct field values at target positions from sensor observations."""

    @abstractmethod
    def reconstruct(
        self,
        *,
        sensor_positions: npt.NDArray[np.float64],
        sensor_values: npt.NDArray[np.float64],
        target_positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Map (sensor, ...) values to (target, ...).

        Positions have shape (sensor, xyz) and (target, xyz) in the same
        coordinate frame and units. Trailing value axes are preserved.
        """


class FieldReconNearest(IFieldRecon):
    """Assign the closest sensor's value using Euclidean distance."""

    __slots__ = ()

    def reconstruct(
        self,
        *,
        sensor_positions: npt.NDArray[np.float64],
        sensor_values: npt.NDArray[np.float64],
        target_positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        _validate_inputs(sensor_positions, sensor_values, target_positions)
        tree = KDTree(sensor_positions, leafsize=16)
        _, nearest_indices = tree.query(target_positions)
        return sensor_values[np.asarray(nearest_indices)]


@dataclass(slots=True, kw_only=True)
class FieldReconLinear(IFieldRecon):
    """Interpolate over a Delaunay triangulation in a plane or volume.

    Sensor sites must span a plane or volume. Targets outside the convex hull
    or off the sensor plane receive `fill_value`, which defaults to NaN.
    """

    fill_value: float = np.nan

    def reconstruct(
        self,
        *,
        sensor_positions: npt.NDArray[np.float64],
        sensor_values: npt.NDArray[np.float64],
        target_positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        _validate_inputs(sensor_positions, sensor_values, target_positions)
        origin = sensor_positions.mean(axis=0)
        centred_sensors = sensor_positions - origin
        _, singular_values, directions = np.linalg.svd(
            centred_sensors, full_matrices=False
        )
        tolerance = (
            singular_values[0] * max(centred_sensors.shape) * np.finfo(np.float64).eps
        )
        dimension = int(np.count_nonzero(singular_values > tolerance))
        if dimension < 2:
            raise ValueError("Linear reconstruction needs non-collinear sensor sites.")

        if dimension == 3:
            interpolate = LinearNDInterpolator(
                centred_sensors, sensor_values, fill_value=self.fill_value
            )
            return np.asarray(interpolate(target_positions - origin))

        basis = directions[:2].T
        centred_targets = target_positions - origin
        off_plane = np.abs(centred_targets @ directions[2]) > tolerance
        interpolate = LinearNDInterpolator(
            centred_sensors @ basis, sensor_values, fill_value=self.fill_value
        )
        predictions = np.asarray(interpolate(centred_targets @ basis))
        predictions[off_plane] = self.fill_value
        return predictions


def _validate_inputs(
    sensor_positions: npt.NDArray[np.float64],
    sensor_values: npt.NDArray[np.float64],
    target_positions: npt.NDArray[np.float64],
) -> None:
    for positions in (sensor_positions, target_positions):
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Positions must have shape (point, 3).")
        if not np.all(np.isfinite(positions)):
            raise ValueError("Positions must be finite.")
    if len(sensor_positions) == 0:
        raise ValueError("Reconstruction needs at least one sensor.")
    if sensor_values.ndim == 0 or sensor_values.shape[0] != len(sensor_positions):
        raise ValueError("The first value axis must match the number of sensors.")
