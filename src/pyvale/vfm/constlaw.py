import enum
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class EIdentificationType(enum.Enum):
    Linear = enum.auto()
    Nonlinear = enum.auto()


class IConstitutiveLaw(ABC):
    @abstractmethod
    def get_identification_type(self) -> EIdentificationType:
        pass

    @abstractmethod
    def calculate_stress(
        self,
        strain: npt.NDArray[np.float64],
        constitutive_parameter_maps: dict[str, npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        pass
