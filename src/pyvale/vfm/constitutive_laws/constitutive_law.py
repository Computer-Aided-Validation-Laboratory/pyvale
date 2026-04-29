from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from pyvale.vfm.parameter import Parameter


class ConstitutiveLaw(ABC):

    @abstractmethod
    def calculate_stress(
        self,
        parameters: dict[str, Parameter],
        strain: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        pass
