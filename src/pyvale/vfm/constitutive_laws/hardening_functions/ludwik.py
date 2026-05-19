from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constitutive_laws.hardening_functions.hardening_function import (
    IHardeningFunction,
)


@dataclass(slots=True)
class LudwikHardening(IHardeningFunction):
    yield_strength_label: str
    strength_coefficient_label: str
    hardening_exponent_label: str

    def __init__(
        self,
        yield_strength_label: str | None = None,
        strength_coefficient_label: str | None = None,
        hardening_exponent_label: str | None = None
    ) -> None:
        if yield_strength_label is not None:
            self.yield_strength_label = yield_strength_label
        else:
            self.yield_strength_label= "yield_strength"

        if strength_coefficient_label is not None:
            self.strength_coefficient_label= strength_coefficient_label
        else:
            self.strength_coefficient_label= "strength_coefficient"

        if hardening_exponent_label is not None:
            self.hardening_exponent_label = hardening_exponent_label
        else:
            self.hardening_exponent_label = "hardening_exponent"

    def hardening(
        self,
        constitutive_parameter_maps: dict[str, npt.NDArray[np.float64]],
        equivalent_plastic_strain: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        yield_strength = constitutive_parameter_maps[
            self.yield_strength_label
        ].ravel()

        strength_coefficient = constitutive_parameter_maps[
            self.strength_coefficient_label
        ].ravel()

        hardening_exponent = constitutive_parameter_maps[
            self.hardening_exponent_label
        ].ravel()

        clamped_equivalent_plastic_strain = np.maximum(
            equivalent_plastic_strain, 1e-14
        )

        yield_stress = (
            yield_strength
            + strength_coefficient
            * clamped_equivalent_plastic_strain**hardening_exponent
        )

        delta_yield_stress = (
            hardening_exponent
            * strength_coefficient
            * (
                clamped_equivalent_plastic_strain
                ** (hardening_exponent - 1)
            )
        )

        return yield_stress, delta_yield_stress
