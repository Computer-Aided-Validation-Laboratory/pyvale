from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constitutive_laws.hardening_functions.hardening_function import (
    IHardeningFunction,
)


@dataclass(slots=True)
class SwiftHardening(IHardeningFunction):
    strength_coefficient_label: str
    strain_offset_label: str
    hardening_exponent_label: str

    def __init__(
        self,
        strength_coefficient_label: str | None = None,
        strain_offset_label: str | None = None,
        hardening_exponent_label: str | None = None
    ) -> None:
        if strength_coefficient_label is not None:
            self.strength_coefficient_label= strength_coefficient_label
        else:
            self.strength_coefficient_label= "strength_coefficient"

        if strain_offset_label is not None:
            self.strain_offset_label = strain_offset_label
        else:
            self.strain_offset_label = "strain_offset"

        if hardening_exponent_label is not None:
            self.hardening_exponent_label = hardening_exponent_label
        else:
            self.hardening_exponent_label = "hardening_exponent"

    def hardening(
        self,
        constitutive_parameter_maps: dict[str, npt.NDArray[np.float64]],
        equivalent_plastic_strain: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        strength_coefficient = constitutive_parameter_maps[
            self.strength_coefficient_label
        ].ravel()

        strain_offset = constitutive_parameter_maps[
            self.strain_offset_label
        ].ravel()

        hardening_exponent = constitutive_parameter_maps[
            self.hardening_exponent_label
        ].ravel()

        strain_term = strain_offset + equivalent_plastic_strain

        yield_stress = (
            strength_coefficient * (strain_term ** hardening_exponent)
        )

        delta_yield_stress = (
            strength_coefficient
            * hardening_exponent
            * strain_term ** (hardening_exponent - 1)
        )

        return yield_stress, delta_yield_stress
