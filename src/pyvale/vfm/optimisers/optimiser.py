from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constitutive_laws.constitutive_law import ConstitutiveLaw
from pyvale.vfm.experiment_data import ExperimentData
from pyvale.vfm.metrics.metric import Metric
from pyvale.vfm.spatial_parameterisations.spatial_parameterisation import (
    SpatialParameterisation,
)


class Optimiser(ABC):
    # Run a set of optimisation passes until a best guess is found
    @abstractmethod
    def optimise(
        self,
        constitutive_law: ConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, SpatialParameterisation],
        weighted_metrics: list[tuple[float, Metric]],
        experiment_data: ExperimentData,
    ) -> None:
        pass
