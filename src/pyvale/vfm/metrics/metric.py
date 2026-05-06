from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

class Metric(ABC):
    @abstractmethod
    def evaluate(
        self,
        stress: npt.NDArray[np.float64]
    ) -> float:
        pass
