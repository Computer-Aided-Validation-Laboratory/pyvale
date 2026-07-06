from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.spatialparam import ISpatialParameterisation


class IMetric(ABC):
    @abstractmethod
    def initialise(
        self,
        experiment_data: ExperimentData
    ) -> None:
        """
        Perform one-off setup for the metric before evaluation,
        any expensive precomputation should be performed here.

        Parameters
        ----------
        experiment_data : ExperimentData
            Measured DIC data
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        stress: npt.NDArray[np.float64],
        constitutive_law: IConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, ISpatialParameterisation],
        experiment_data: ExperimentData
    ) -> npt.NDArray[np.float64]:
        pass
