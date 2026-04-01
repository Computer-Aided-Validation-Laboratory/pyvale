import numpy as np
import numpy.typing as npt

from pyvale.vfm.mechanical_properties import (
    EConstituitiveLaw,
    EParameterLabel,
    MechanicalProperties,
)


def hardening(
    constituitive_law: EConstituitiveLaw,
    equivalent_plastic_strain: npt.NDArray[np.float64],
    mechanical_properties: MechanicalProperties,
    size_x: int,
    size_y: int,
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

    parameters = mechanical_properties.parameters

    match constituitive_law:
        case EConstituitiveLaw.LinearHardening:
            yield_strength = (
                parameters[EParameterLabel.YieldStrength].to_map(size_x, size_y)
            )
            hardening_modulus = (
                parameters[EParameterLabel.HardeningModulus].to_map(
                    size_x, size_y
                )
            )

            yield_strength_flat = yield_strength.ravel()
            hardening_modulus_flat = hardening_modulus.ravel()

            yield_stress = yield_strength_flat + (
                hardening_modulus_flat * equivalent_plastic_strain
            )

            return yield_stress, hardening_modulus_flat

        case EConstituitiveLaw.SwiftHardening:
            strength_coefficient = (
                parameters[EParameterLabel.StrengthCoefficient].to_map(
                    size_x, size_y
                )
            )
            strain_offset = (
                parameters[EParameterLabel.StrainOffset].to_map(size_x, size_y)
            )
            hardening_exponent = (
                parameters[EParameterLabel.HardeningExponent].to_map(
                    size_x, size_y
                )
            )

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

        case EConstituitiveLaw.VoceHardening:
            yield_strength = (
                parameters[EParameterLabel.YieldStrength].to_map(size_x, size_y)
            )
            hardening_modulus = (
                parameters[EParameterLabel.HardeningModulus].to_map(
                    size_x, size_y
                )
            )
            saturation_stress = (
                parameters[EParameterLabel.SaturationStress].to_map(
                    size_x, size_y
                )
            )
            rate_parameter = (
                parameters[EParameterLabel.RateParameter].to_map(size_x, size_y)
            )

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

        case EConstituitiveLaw.LudwikHardening:
            yield_strength = (
                parameters[EParameterLabel.YieldStrength].to_map(size_x, size_y)
            )
            strength_coefficient = (
                parameters[EParameterLabel.StrengthCoefficient].to_map(
                    size_x, size_y
                )
            )
            hardening_exponent = (
                parameters[EParameterLabel.HardeningExponent].to_map(
                    size_x, size_y
                )
            )

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
            variants = ", ".join(e.name for e in EConstituitiveLaw) 

            raise NotImplementedError(
                f"Hardening law '{constituitive_law}' is not yet implemented. "
                f"Supported laws: {variants}"
            )


