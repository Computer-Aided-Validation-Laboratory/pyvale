import numpy as np

from pyvale.vfm.constlaw import EIdentificationType
from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.validation import (
    validate_experiment_data,
    validate_identification_config,
)
from pyvale.vfm.identificationconfig import IdentificationConfig
from pyvale.vfm.spatialparam import (
    evaluate_parameterisations_to_map,
    initialise_parameterisations_from_constitutive_parameter,
)


def run_identification(
    experiment_data: ExperimentData,
    identification_config: IdentificationConfig
) -> dict[str, ConstitutiveParameter]:
    validate_experiment_data(experiment_data)
    validate_identification_config(identification_config)

    match identification_config.constitutive_law.get_identification_type():
        # TODO: implement linear case
        case EIdentificationType.Linear:
            ...
        case EIdentificationType.Nonlinear:
            parameter_map_size = np.array(
                experiment_data.specimen_geometry.x.shape,
                dtype=np.uint32
            )

            for phase in identification_config.phases:
                # Initialise spatial parameterisation from constitutive
                # parameter maps
                for param_name, sps in phase.spatial_parameterisations.items():
                    initialise_parameterisations_from_constitutive_parameter(
                        sps,
                        identification_config.parameters[param_name],
                        parameter_map_size,
                    )

                for metric in phase.metrics:
                    metric.initialise(experiment_data)

                # Optimise the degrees of freedom of the spatial
                # parameterisations
                opt_spatial_parameterisations = phase.optimiser.optimise(
                    identification_config.constitutive_law,
                    parameter_map_size,
                    phase.spatial_parameterisations,
                    phase.metrics,
                    phase.objective_function,
                    experiment_data
                )

                # Update constitutive parameter maps from optimised spatial
                # parameterisations
                for param_name, sps in opt_spatial_parameterisations.items():
                    identification_config.parameters[param_name].map = (
                        evaluate_parameterisations_to_map(
                            sps,
                            parameter_map_size
                        )
                    )

    return identification_config.parameters
