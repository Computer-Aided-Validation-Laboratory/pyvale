from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class ObjectiveResult(ABC):
    scalar: float | None
    residuals: npt.NDArray[np.float64] | None
    # TODO: needed for multi objective optimisation I think
    # objectives: npt.NDArray[np.float64] | None


# TODO: no idea what a good name for this is
class Objective(ABC):
    @abstractmethod
    def evaluate(
        self,
        metric_result: npt.NDArray[np.float64],
    ) -> ObjectiveResult:
        pass
