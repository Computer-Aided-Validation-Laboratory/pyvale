from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.spatialparam import ISpatialParameterisation


@dataclass(slots=True)
class MetricResult:
    residual: npt.NDArray[np.float64] | None = None
    additional_fields: dict | None = None


class IMetric(ABC):
    """
    Interface (abstract base class) for a virtual-work metric.

    A metric evaluates the discrepancy between a candidate stress field and
    the measured boundary conditions using the virtual work principle.
    Multiple metrics (e.g. for different virtual fields) may be
    combined in an objective function
    """

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
        spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
        experiment_data: ExperimentData,
    ) -> MetricResult:
        """
        Evaluate the metric for a given stress candidate.

        Parameters
        ----------
        stress : npt.NDArray[np.float64]
            Candidate stress field, shape ``(timesteps, components, y, x)``
        constitutive_law : IConstitutiveLaw
            Constitutive law used to produce the stress
        parameter_map_size : npt.NDArray[np.uint32]
            Spatial dimensions ``(y, x)`` of the parameter maps
        spatial_parameterisations : dict[str, list[ISpatialParameterisation]]
            Current spatial parameterisations keyed by parameter name
        experiment_data : ExperimentData
            Measured DIC data

        Returns
        -------
        npt.NDArray[np.float64]
            Metric value(s) per timestep or per spatial point
        """
        pass
