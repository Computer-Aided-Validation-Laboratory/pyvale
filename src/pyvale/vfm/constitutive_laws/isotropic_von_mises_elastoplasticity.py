from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constitutive_laws.constitutive_law import (
    EIdentificationType,
    IConstitutiveLaw,
)
from pyvale.vfm.constitutive_laws.hardening_functions.hardening_function import (
    IHardeningFunction,
)
from pyvale.vfm.constitutive_laws.radial_return import radial_return


@dataclass(slots=True)
class IsotropicVonMisesElastoplasticity(IConstitutiveLaw):
    hardening_function: IHardeningFunction
    elastic_modulus_label: str
    poissons_ratio_label: str

    def __init__(
        self,
        hardening_function: IHardeningFunction,
        elastic_modulus_label: str | None = None,
        poissons_ratio_label: str | None = None
    ) -> None:
        self.hardening_function = hardening_function

        if elastic_modulus_label is not None:
            self.elastic_modulus_label = elastic_modulus_label
        else:
            self.elastic_modulus_label = "elastic_modulus"

        if poissons_ratio_label is not None:
            self.poissons_ratio_label = poissons_ratio_label
        else:
            self.poissons_ratio_label = "poissons_ratio"

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
            constitutive_parameter_maps[self.elastic_modulus_label],
            constitutive_parameter_maps[self.poissons_ratio_label],
            self.hardening_function
        )

        return stress
