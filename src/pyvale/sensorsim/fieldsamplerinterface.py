# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Interfaces for sampling physical fields in world coordinates."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

FieldFunc = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64]],
    npt.NDArray[np.float64],
]


class IFieldSampler(ABC):
    """Sample a field at XYZ points and times, in world coordinates."""

    @abstractmethod
    def get_time_steps(self) -> npt.NDArray[np.float64]:
        """Return the default sampling times."""

    @abstractmethod
    def get_num_components(self) -> int:
        """Return the size of the sampler's component axis."""

    @abstractmethod
    def sample(
        self,
        points: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Return values shaped (point, component, time)."""


def interp_to_sample_time(
    sample_at_sim_time: npt.NDArray[np.float64],
    sim_time_steps: npt.NDArray[np.float64],
    sample_times: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Interpolate spatially sampled values onto the requested time grid.

    Parameters
    ----------
    sample_at_sim_time : numpy.ndarray
        Values shaped (sensor, component, simulation time).
    sim_time_steps : numpy.ndarray
        Time coordinates for the last axis of sample_at_sim_time.
    sample_times : numpy.ndarray
        Requested sampling times, in the same units as sim_time_steps.

    Returns
    -------
    numpy.ndarray
        Values shaped (sensor, component, requested time).
    """

    def interp_trace(trace: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.interp(sample_times, sim_time_steps, trace)

    num_sensors, num_components, _ = sample_at_sim_time.shape
    sampled = np.empty((num_sensors, num_components, len(sample_times)))

    for component_index in range(num_components):
        traces = sample_at_sim_time[:, component_index, :]
        sampled[:, component_index, :] = np.apply_along_axis(interp_trace, -1, traces)

    return sampled
