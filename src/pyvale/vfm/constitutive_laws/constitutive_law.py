from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from pyvale.vfm.identification import EIdentificationType
from pyvale.vfm.parameter import ConstitutiveParameter


class ConstitutiveLaw(ABC):
    @property
    @abstractmethod
    def identification_type(self) -> EIdentificationType:
        pass

    @abstractmethod
    def calculate_stress(
        self,
        parameters: dict[str, ConstitutiveParameter],
        strain: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        pass
