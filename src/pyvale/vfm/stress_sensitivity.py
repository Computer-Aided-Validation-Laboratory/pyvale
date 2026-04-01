from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.mechanical_properties import (
    EDOFLabel,
    EParameterLabel,
    MechanicalProperties,
    UnknownParameter,
)
from pyvale.vfm.radial_return import radial_return


@dataclass(slots=True)
class StressSensitivity:
    total: npt.NDArray[np.float64]
    incremental: npt.NDArray[np.float64]


# TODO: do we need normalisation here?
def calculate_stress_sensitivity(
    stress_reference: npt.NDArray[np.float64],
    strain: npt.NDArray[np.float64],
    mechanical_properties: MechanicalProperties,
    timestep_deltas: npt.NDArray[np.float64],
    perturbation_factor: float = 0.15
) ->  dict[
    EParameterLabel,
    list[dict[EDOFLabel, StressSensitivity]]
]:
    unknown_params = mechanical_properties.get_unknown_parameters()

    dofs = {
        label: param.get_degrees_of_freedom()
        for label, param in unknown_params.items()
    }

    stress_sensitivities = {}

    for param_label, parameterisations in dofs.items():

        per_parameterisation_ss = []

        for i, p in enumerate(parameterisations):

            per_dof_ss = {}

            for dof_label, dof in p.items():
                perturbed_dof = dof.value * (1 - perturbation_factor)

                perturbed_props = deepcopy(mechanical_properties)

                match perturbed_props.parameters[param_label]:
                    case UnknownParameter(_, _, parameterisation):
                        parameterisation[i].update_degree_of_freedom_value(
                            dof_label,
                            perturbed_dof
                        )
                    case other:
                        raise TypeError(
                            f"Unexpected type: {type(other).__name__}"
                        )

                (perturbed_stress, _, _, _) = radial_return(
                    strain, perturbed_props
                )

                total_stress_sensitivity  = stress_reference - perturbed_stress

                incremental_stress_sensitivity = np.zeros_like(stress_reference)
                incremental_stress_sensitivity[:, :, :, 1:] = (
                    total_stress_sensitivity[:, :, :, 1:]
                    - total_stress_sensitivity[:, :, :, 0:-1]
                )

                time_normalisation_mask = np.reshape(
                    timestep_deltas, (1, 1, 1, timestep_deltas.size)
                )

                incremental_stress_sensitivity = (
                    incremental_stress_sensitivity / time_normalisation_mask
                )

                per_dof_ss[dof_label] = StressSensitivity(
                    total_stress_sensitivity, incremental_stress_sensitivity
                )

            per_parameterisation_ss.append(per_dof_ss)

        stress_sensitivities[param_label] = per_parameterisation_ss

    return stress_sensitivities
