from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

class Optimiser(ABC):

    # Run a set of optimisation passes until a best guess is found
    @abstractmethod
    def optimise(self) -> None:
        pass
