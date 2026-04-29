from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

class Parameterisation(ABC):

    @abstractmethod
    def to_map(self, size_y: int, size_x: int) -> npt.NDArray[np.float64]:
        pass
