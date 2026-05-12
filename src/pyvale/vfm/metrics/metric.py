from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from pyvale.vfm.experiment_data import ExperimentData
from pyvale.vfm.spatial_parameterisations.spatial_parameterisation import (
    SpatialParameterisation,
)


class Metric(ABC):
    @abstractmethod
    def evaluate(
        self,
        stress: npt.NDArray[np.float64],
        spatial_parameterisations: dict[str, SpatialParameterisation],
        experiment_data: ExperimentData
    ) -> float:
        pass
