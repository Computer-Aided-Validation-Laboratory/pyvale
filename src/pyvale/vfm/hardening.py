from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


class IHardeningFunction(ABC):
    """
    Interface (abstract base class) for an isotropic hardening law.

    A hardening law maps the current constitutive parameters and the
    equivalent plastic strain onto the current yield stress and its slope,
    and declares which constitutive parameters it requires
    """

    @abstractmethod
    def hardening(
        self,
        constitutive_parameter_maps: dict[str, npt.NDArray[np.float64]],
        equivalent_plastic_strain: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Return the current yield stress and its derivative for this
        hardening law.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple ``(yield_stress, hardening_slope)`` where ``yield_stress``
            is the current yield stress at all datapoints and
            ``hardening_slope`` is the derivative of the yield stress with
            respect to the equivalent plastic strain (the hardening slope)
        """
        pass

    @abstractmethod
    def get_required_parameters(self) -> list[str]:
        """
        Return the list of constitutive parameters required for
        this hardening law.

        Returns
        -------
        list[str]
            All parameter name strings this hardening law requires
        """
        pass


@dataclass(slots=True)
class HardeningLinear(IHardeningFunction):
    """
    Linear (bilinear) isotropic hardening.

    The yield stress grows linearly with equivalent plastic strain:
    ``sigma_y = yield_strength + hardening_modulus * eps_p``.

    Requires the ``yield_strength`` and ``hardening_modulus`` parameters. The
    label arguments allow these to be renamed if your parameter dictionary
    uses different keys
    """

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

    def get_required_parameters(self) -> list[str]:
        return [self.yield_strength_label, self.hardening_modulus_label]

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


@dataclass(slots=True)
class HardeningSwift(IHardeningFunction):
    """
    Swift power-law isotropic hardening.

    The yield stress follows
    ``sigma_y = strength_coefficient * (strain_offset + eps_p) ** hardening_exponent``.

    Requires the ``strength_coefficient``, ``strain_offset`` and
    ``hardening_exponent`` parameters. The label arguments allow these to be
    renamed if your parameter dictionary uses different keys
    """

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

    def get_required_parameters(self) -> list[str]:
        return [
            self.strength_coefficient_label,
            self.strain_offset_label,
            self.hardening_exponent_label,
        ]

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


@dataclass(slots=True)
class HardeningVoce(IHardeningFunction):
    """
    Voce (saturating) isotropic hardening with a linear term.

    Combines a linear hardening term with an exponentially saturating term:
    ``sigma_y = yield_strength + hardening_modulus * eps_p
    + saturation_stress * (1 - exp(-rate_parameter * eps_p))``.

    Requires the ``yield_strength``, ``hardening_modulus``,
    ``saturation_stress`` and ``rate_parameter`` parameters. The label
    arguments allow these to be renamed if your parameter dictionary uses
    different keys
    """

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

    def get_required_parameters(self) -> list[str]:
        return [
            self.yield_strength_label,
            self.hardening_modulus_label,
            self.saturation_stress_label,
            self.rate_parameter_label,
        ]

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


@dataclass(slots=True)
class HardeningLudwik(IHardeningFunction):
    """
    Ludwik power-law isotropic hardening.

    The yield stress follows
    ``sigma_y = yield_strength + strength_coefficient * eps_p ** hardening_exponent``.

    Requires the ``yield_strength``, ``strength_coefficient`` and
    ``hardening_exponent`` parameters. The label arguments allow these to be
    renamed if your parameter dictionary uses different keys
    """

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

    def get_required_parameters(self) -> list[str]:
        return [
            self.yield_strength_label,
            self.strength_coefficient_label,
            self.hardening_exponent_label,
        ]

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
