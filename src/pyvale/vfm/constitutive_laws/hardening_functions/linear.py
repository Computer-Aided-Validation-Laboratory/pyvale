from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constitutive_laws.hardening_functions.hardening_function import (
    IHardeningFunction,
)


@dataclass(slots=True)
class LinearHardening(IHardeningFunction):
    yield_strength_label: str
    hardening_modulus_label: str

    def __init__(
        self,
        yield_strength_label: str | None = None,
        hardening_modulus_label: str | None = None
    ) -> None:
        if yield_strength_label is not None:
            self.yield_strength_label = yield_strength_label
        else:
            self.yield_strength_label= "yield_strength"

        if hardening_modulus_label is not None:
            self.hardening_modulus_label = hardening_modulus_label
        else:
            self.hardening_modulus_label = "hardening_modulus"

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

        yield_stress = yield_strength + (
        hardening_modulus * equivalent_plastic_strain
        )

        return yield_stress, hardening_modulus
