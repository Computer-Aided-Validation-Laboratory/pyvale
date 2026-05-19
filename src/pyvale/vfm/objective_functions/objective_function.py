from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class IScalarObjectiveFunction(ABC):
    @abstractmethod
    def evaluate(
        self,
        metric_results: list[npt.NDArray[np.float64]],
    ) -> float:
        pass


class IVectorObjectiveFunction(ABC):
    @abstractmethod
    def evaluate(
        self,
        metric_results: list[npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        pass


IObjectiveFunction = IScalarObjectiveFunction | IVectorObjectiveFunction
