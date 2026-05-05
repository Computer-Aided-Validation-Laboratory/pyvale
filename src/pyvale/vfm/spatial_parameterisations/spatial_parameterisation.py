from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from pyvale.vfm.parameter import ConstitutiveParameter


class SpatialParameterisation(ABC):

    @abstractmethod
    def to_map(
        self,
        params: dict[str, ConstitutiveParameter],
        size: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.float64]:
        pass
