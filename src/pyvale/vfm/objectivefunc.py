from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class IScalarObjectiveFunction(ABC):
    """
    Interface (abstract base class) for a scalar objective function.

    Aggregates a list of metric results into a single scalar value that the
    optimiser minimises
    """

    @abstractmethod
    def evaluate(
        self,
        metric_results: list[npt.NDArray[np.float64]],
    ) -> float:
        """
        Aggregate metric results into a scalar cost

        Parameters
        ----------
        metric_results : list[npt.NDArray[np.float64]]
            One array per metric, each with the metric's output

        Returns
        -------
        float
            Scalar objective value to minimise
        """
        pass


class IVectorObjectiveFunction(ABC):
    """
    Interface (abstract base class) for a vector objective function.

    Aggregates metric results into a vector that the optimiser minimises
    """

    @abstractmethod
    def evaluate(
        self,
        metric_results: list[npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        """
        Aggregate metric results into a residual vector

        Parameters
        ----------
        metric_results : list[npt.NDArray[np.float64]]
            One array per metric, each with the metric's output

        Returns
        -------
        npt.NDArray[np.float64]
            Residual vector for the optimiser
        """
        pass


IObjectiveFunction = IScalarObjectiveFunction | IVectorObjectiveFunction
"""Union type for objective functions that produce a scalar or vector cost"""
