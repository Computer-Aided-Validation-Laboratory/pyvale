import enum

import numpy as np
import numpy.typing as npt

class EHardening(enum.Enum):
    Linear = enum.auto()
    Swift = enum.auto()
    Voce = enum.auto()
    Ludwik = enum.auto()


def hardening_func(
    hardening: EHardening,
    parameter_maps: dict[str, npt.NDArray[np.float64]],
    equivalent_plastic_strain: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Return current yield stress and its derivative for the active hardening law.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        yield_stress : Current yield stress at all datapoints
        delta_yield_stress_delta_equivalent_plastic_strain : Derivative of
            yield stress with respect to equivalent plastic strain
            (i.e. hardening slope)
    """

    match hardening:
        case EHardening.Linear:
            yield_strength = parameter_maps["yield_strength"]
            hardening_modulus = parameter_maps["hardening_modulus"]

            yield_strength_flat = yield_strength.ravel()
            hardening_modulus_flat = hardening_modulus.ravel()

            yield_stress = yield_strength_flat + (
                hardening_modulus_flat * equivalent_plastic_strain
            )

            return yield_stress, hardening_modulus_flat

        case EHardening.Swift:
            strength_coefficient = parameter_maps["strength_coefficient"]
            strain_offset = parameter_maps["strain_offset"]
            hardening_exponent = parameter_maps["hardening_exponent"]

            strength_coefficient_flat = strength_coefficient.ravel()
            strain_offset_flat = strain_offset.ravel()
            hardening_exponent_flat = hardening_exponent.ravel()

            strain_term = strain_offset_flat + equivalent_plastic_strain

            yield_stress = (
                strength_coefficient_flat * strain_term**hardening_exponent_flat
            )

            delta_yield_stress = (
                strength_coefficient_flat
                * hardening_exponent_flat
                * strain_term ** (hardening_exponent_flat - 1)
            )

            return yield_stress, delta_yield_stress

        case EHardening.Voce:
            yield_strength = parameter_maps["yield_strength"]
            hardening_modulus = parameter_maps["hardening_modulus"]
            saturation_stress = parameter_maps["saturation_stress"]
            rate_parameter = parameter_maps["rate_parameter"]

            yield_strength_flat = yield_strength.ravel()
            hardening_modulus_flat = hardening_modulus.ravel()
            saturation_stress_flat = saturation_stress.ravel()
            rate_parameter_flat = rate_parameter.ravel()

            exp_term = np.exp(-rate_parameter_flat * equivalent_plastic_strain)

            yield_stress = (
                yield_strength_flat
                + hardening_modulus_flat * equivalent_plastic_strain
                + saturation_stress_flat * (1 - exp_term)
            )

            delta_yield_stress = (
                hardening_modulus_flat
                + saturation_stress_flat * rate_parameter_flat * exp_term
            )

            return yield_stress, delta_yield_stress

        case EHardening.Ludwik:
            yield_strength = parameter_maps["yield_strength"]
            strength_coefficient = parameter_maps["strength_coefficient"]
            hardening_exponent = parameter_maps["hardening_exponent"]

            yield_strength_flat = yield_strength.ravel()
            strength_coefficient_flat = strength_coefficient.ravel()
            hardening_exponent_flat = hardening_exponent.ravel()

            clamped_equivalent_plastic_strain = np.maximum(
                equivalent_plastic_strain, 1e-14
            )

            yield_stress = (
                yield_strength_flat
                + strength_coefficient_flat
                * clamped_equivalent_plastic_strain**hardening_exponent_flat
            )

            delta_yield_stress = (
                hardening_exponent_flat
                * strength_coefficient_flat
                * (
                    clamped_equivalent_plastic_strain
                    ** (hardening_exponent_flat - 1)
                )
            )

            return yield_stress, delta_yield_stress

        case _:
            raise NotImplementedError(
                f"Unsupported hardening: '{hardening}'"
            )


