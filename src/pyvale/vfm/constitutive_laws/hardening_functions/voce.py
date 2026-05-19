from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constitutive_laws.hardening_functions.hardening_function import (
    IHardeningFunction,
)


@dataclass(slots=True)
class VoceHardening(IHardeningFunction):
    yield_strength_label: str
    hardening_modulus_label: str
    saturation_stress_label: str
    rate_parameter_label: str

    def __init__(
        self,
        yield_strength_label: str | None = None,
        hardening_modulus_label: str | None = None,
        saturation_stress_label: str | None = None,
        rate_parameter_label: str | None = None
    ) -> None:
        if yield_strength_label is not None:
            self.yield_strength_label = yield_strength_label
        else:
            self.yield_strength_label= "yield_strength"

        if hardening_modulus_label is not None:
            self.hardening_modulus_label = hardening_modulus_label
        else:
            self.hardening_modulus_label = "hardening_modulus"

        if saturation_stress_label is not None:
            self.saturation_stress_label = saturation_stress_label
        else:
            self.saturation_stress_label = "saturation_stress"

        if rate_parameter_label is not None:
            self.rate_parameter_label = rate_parameter_label
        else:
            self.rate_parameter_label = "rate_parameter"

    def hardening(
        self,
        constitutive_parameter_maps: dict[str, npt.NDArray[np.float64]],
        equivalent_plastic_strain: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        yield_strength = constitutive_parameter_maps[
            self.yield_strength_label
        ].ravel()

        hardening_modulus = constitutive_parameter_maps[
            self.hardening_modulus_label
        ].ravel()

        saturation_stress = constitutive_parameter_maps[
            self.saturation_stress_label
        ].ravel()

        rate_parameter = constitutive_parameter_maps[
            self.rate_parameter_label
        ].ravel()

        exp_term = np.exp(-rate_parameter * equivalent_plastic_strain)

        yield_stress = (
            yield_strength
            + hardening_modulus * equivalent_plastic_strain
            + saturation_stress * (1 - exp_term)
        )

        delta_yield_stress = (
            hardening_modulus
            + saturation_stress * rate_parameter * exp_term
        )

        return yield_stress, delta_yield_stress
