from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constlaw import (
    EIdentificationType,
    IConstitutiveLaw,
)
from pyvale.vfm.hardening import IHardeningFunction
from pyvale.vfm.radialreturn import radial_return


@dataclass(slots=True)
class IsotropicVonMisesElastoplasticity(IConstitutiveLaw):
    """
    Isotropic von Mises (J2) elasto-plasticity in plane stress.

    Combines linear isotropic elasticity with a J2 yield surface and the
    supplied isotropic ``hardening_function``. The required parameters are
    ``elastic_modulus`` and ``poissons_ratio`` plus whichever parameters the
    hardening law needs. The label arguments allow the elastic parameters to
    be renamed if your parameter dictionary uses different keys
    """

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

    def get_identification_type(self) -> EIdentificationType:
        return EIdentificationType.Nonlinear

    def get_required_parameters(self) -> list[str]:
        params = [self.elastic_modulus_label, self.poissons_ratio_label]
        params.extend(self.hardening_function.get_required_parameters())
        return params

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
