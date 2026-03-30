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
class StressSensitivities:
    total: dict[
        EParameterLabel,
        list[dict[EDOFLabel, npt.NDArray[np.float64]]]
    ]
    incremental: dict[
        EParameterLabel,
        list[dict[EDOFLabel, npt.NDArray[np.float64]]]
    ]


# TODO: do we need normalisation here?
def calculate_stress_sensitivity(
    stress_reference: npt.NDArray[np.float64],
    strain: npt.NDArray[np.float64],
    mechanical_properties: MechanicalProperties,
    timestep_deltas: npt.NDArray[np.float64],
    perturbation_factor: float = 0.15
) -> StressSensitivities:
    unknown_params = mechanical_properties.get_unknown_parameters()

    dofs = {
        label: param.get_degrees_of_freedom()
        for label, param in unknown_params.items()
    }

    total_stress_sensitivities = {}
    incremental_stress_sensitivities = {}

    for param_label, parameterisations in dofs.items():

        per_parameterisation_tss = []
        per_parameterisation_iss = []

        for i, p in enumerate(parameterisations):

            per_dof_tss = {}
            per_dof_iss = {}

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

                # TODO: use updated radial reuturn func once it's pulled in
                perturbed_stress = radial_return(strain, perturbed_props)

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

                per_dof_tss[dof_label] = total_stress_sensitivity
                per_dof_iss[dof_label] = incremental_stress_sensitivity

            per_parameterisation_tss.append(per_dof_tss)
            per_parameterisation_iss.append(per_dof_iss)

        total_stress_sensitivities[param_label] = per_parameterisation_tss
        incremental_stress_sensitivities[param_label] = per_parameterisation_iss

    return StressSensitivities(
        total_stress_sensitivities,
        incremental_stress_sensitivities
    )
