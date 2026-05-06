from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constitutive_laws.constitutive_law import ConstitutiveLaw
from pyvale.vfm.identification import EIdentificationType
from pyvale.vfm.parameter import ConstitutiveParameter
from pyvale.vfm.radial_return import radial_return

@dataclass(slots=True)
class LinearHardening(ConstitutiveLaw):
    @property
    def identification_type(self) -> EIdentificationType:
        return EIdentificationType.Nonlinear

    def calculate_stress(
      self,
      parameters: dict[str, ConstitutiveParameter],
      strain: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        # radial_return(strain)
        ...
