from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class ScalarObjectiveFunction(ABC):
    @abstractmethod
    def evaluate(
        self,
        metric_results: list[npt.NDArray[np.float64]],
    ) -> float:
        pass


class VectorObjectiveFunction(ABC):
    @abstractmethod
    def evaluate(
        self,
        metric_results: list[npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        pass


ObjectiveFunction = ScalarObjectiveFunction | VectorObjectiveFunction
