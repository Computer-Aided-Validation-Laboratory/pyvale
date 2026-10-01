from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from pyvale.vfm.metric import MetricResult


class IVectorObjectiveFunction(ABC):
    """
    Interface (abstract base class) for a vector objective function.

    Aggregates metric results into a vector that the optimiser minimises
    """

    @abstractmethod
    def evaluate(
        self,
        metric_results: list[MetricResult],
    ) -> npt.NDArray[np.float64]:
        """
        Aggregate metric results into a residual vector.

        Parameters
        ----------
        metric_results : list[MetricResult]
            One array per metric, each with the metric's output

        Returns
        -------
        npt.NDArray[np.float64]
            Residual vector for the optimiser
        """
        pass


IObjectiveFunction = IVectorObjectiveFunction
"""Alias for the vector objective-function interface."""
