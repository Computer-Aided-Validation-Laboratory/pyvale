from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constitutive_laws.constitutive_law import ConstitutiveLaw
from pyvale.vfm.identification import EIdentificationType
from pyvale.vfm.constitutive_laws.radial_return import radial_return

@dataclass(slots=True)
class LinearHardening(ConstitutiveLaw):
    @property
    def identification_type(self) -> EIdentificationType:
        return EIdentificationType.Nonlinear

    def calculate_stress(
        self,
        strain: npt.NDArray[np.float64],
        constitutive_parameter_maps: dict[str, npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        # radial_return(strain)
        ...
