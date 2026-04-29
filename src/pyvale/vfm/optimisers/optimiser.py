from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

class Optimiser(ABC):

    @abstractmethod
    def optimise(self) -> None:
        pass
