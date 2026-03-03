from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.dic_config import DICConfig
from pyvale.vfm.mechanical_properties import MechanicalProperties
from pyvale.vfm.radial_return import radial_return
from pyvale.vfm.stress import convert_stress_to_4d


@dataclass(slots=True)
class StressSensitivity:
    total: dict[str, npt.NDArray[np.float64]]
    incremental: dict[str, npt.NDArray[np.float64]]

# To compute stress sensitivity we need to
# - take the original stress we calculated
# - perform some normalisation on the degrees of freedom
# - for each degree of freedom
#   - calculate the perturbed stress from a slightly perturbed parameter map on that degree of freedom
#   - calculate their differences
#   - store the total stress sensitivity
#   - store the stress sensitivity difference across timesteps
# Should take as input:
# - a referene stress to compare against
#   - the format seems to be pixels_y x pixels_x x 3 components x 23 timesteps
#   - TODO: where does this format come from, should we continue to use it?
# - strain
# - material properties containing parameter maps
#   - hardening modulus
#   - yield strength
# Should return:
# - total stress sensitivity
# - incremental stress sensitivity
# Assumptions:
# - 2 degrees of freedom
#   - hardening modulus
#   - yield strength
# - linear scaling when purturbing params (what does this mean physically?)
# TODO: arg/return types
# TODO: should output be a list of sensitivities? or a tuple/dict with labels?
# TODO: add normalisation
def calculate_stress_sensitivity(
    stress_reference: npt.NDArray[np.float64],
    strain: npt.NDArray[np.float64],
    mechanical_properties: MechanicalProperties,
    dic_config: DICConfig
) -> StressSensitivity:
    stress_sensitivity = StressSensitivity({}, {})

    # TODO: should this be and input, and what will change this value if so?
    perturbation_factor = 0.15

    # TODO: this should be taken from args somehow, should we take in a dict of label/enum to param map?
    num_degrees_of_freedom = 2

    for i in range(num_degrees_of_freedom):
        perturbed_material_properties = deepcopy(mechanical_properties)

        # TODO: should this be an enum/fetched from dict label?
        if i == 0:
            perturbed_material_properties.yield_strength = (
                mechanical_properties.yield_strength * (1 - perturbation_factor)
            )
        elif i == 1:
            perturbed_material_properties.hardening_modulus = (
                mechanical_properties.hardening_modulus * (1 - perturbation_factor)
            )

        perturbed_stress = radial_return(strain, perturbed_material_properties)
        perturbed_stress = convert_stress_to_4d(perturbed_stress, dic_config)

        total_stress_sensitivity = stress_reference - perturbed_stress

        incremental_stress_sensitivity = np.zeros_like(stress_reference)
        incremental_stress_sensitivity[:, :, :, 1:] = (
            total_stress_sensitivity[:, :, :, 1:]
            - total_stress_sensitivity[:, :, :, 0:-1]
        )

        timestep_deltas = dic_config.calculate_timestep_deltas()
        time_normalisation_mask = np.reshape(timestep_deltas, (1, 1, 1, timestep_deltas.size))

        incremental_stress_sensitivity = incremental_stress_sensitivity / time_normalisation_mask

        if i == 0:
            stress_sensitivity.total["yield_strength"] = total_stress_sensitivity
            stress_sensitivity.incremental["yield_strength"] = incremental_stress_sensitivity
        elif i == 1:
            stress_sensitivity.total["hardening_modulus"] = total_stress_sensitivity
            stress_sensitivity.incremental["hardening_modulus"] = incremental_stress_sensitivity
    
    return stress_sensitivity


