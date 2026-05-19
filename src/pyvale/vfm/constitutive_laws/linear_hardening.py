from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constitutive_laws.constitutive_law import (
    IConstitutiveLaw,
    EIdentificationType,
)
from pyvale.vfm.constitutive_laws.hardening import EHardening
from pyvale.vfm.constitutive_laws.radial_return import radial_return


@dataclass(slots=True)
class LinearHardening(IConstitutiveLaw):
    @property
    def identification_type(self) -> EIdentificationType:
        return EIdentificationType.Nonlinear

    def calculate_stress(
        self,
        strain: npt.NDArray[np.float64],
        constitutive_parameter_maps: dict[str, npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        stress, _, _, _ = radial_return(
            strain,
            constitutive_parameter_maps,
            EHardening.Linear
        )

        return stress
