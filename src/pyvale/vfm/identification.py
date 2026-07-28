import numpy as np

from pyvale.vfm.constlaw import EIdentificationType
from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.identificationconfig import IdentificationConfig
from pyvale.vfm.identificationconfig import IdentificationPhase
from pyvale.vfm.validation import (
    validate_slicewise_independent_phase,
    validate_experiment_data,
    validate_identification_config,
)
from pyvale.vfm.spatialparam import (
    PhaseSpatialState,
    evaluate_parameterisations_to_map,
)


def prepare_phase_runtime(
    phase: IdentificationPhase,
    experiment_data: ExperimentData,
) -> PhaseSpatialState:
    """Prepare one phase runtime once experiment data are available.

    Validation has already checked that the phase configuration is legal.
    This step only resolves data-dependent runtime state such as shared
    supports and metric operators.
    """

    phase_spatial_state = PhaseSpatialState(
        phase.spatial_parameterisations
    )
    phase_spatial_state.prepare(experiment_data)

    for metric in phase.metrics:
        metric.initialise(experiment_data)

    return phase_spatial_state


def run_identification(
    experiment_data: ExperimentData,
    identification_config: IdentificationConfig
) -> dict[str, ConstitutiveParameter]:
    validate_experiment_data(experiment_data)
    validate_identification_config(identification_config)

    for phase_index, phase in enumerate(identification_config.phases):
        validate_slicewise_independent_phase(
            phase,
            phase_index,
        )

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
                phase_spatial_state = prepare_phase_runtime(
                    phase,
                    experiment_data,
                )
                phase_spatial_state.initialise_from_constitutive_parameters(
                    identification_config.parameters,
                    parameter_map_size,
                )

                # Optimise the degrees of freedom of the spatial
                # parameterisations
                opt_spatial_parameterisations = phase.optimiser.optimise(
                    identification_config.constitutive_law,
                    parameter_map_size,
                    phase_spatial_state.spatial_parameterisations,
                    phase.metrics,
                    phase.objective_function,
                    experiment_data
                )

                # Optionally perform post-optimisation refinement of spatial
                # parameterisation degrees of freedom on any spatial
                # parameterisations that support it
                for sps in phase.spatial_parameterisations.values():
                    for sp in sps:
                        if sp.should_perform_refinement():
                            sp.perform_refinement()

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
