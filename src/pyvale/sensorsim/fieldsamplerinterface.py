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
