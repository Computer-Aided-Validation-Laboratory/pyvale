from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pyvale.vfm.identification_linear import run_linear_identification
from pyvale.vfm.identification_nonlinear import run_nonlinear_identification
from pyvale.vfm.mat_to_py_data_parser import load_parsed_test_data
from pyvale.vfm.mechanical_properties import (
    ConstituitiveLaw,
    KnownParameter,
    MechanicalProperties,
    required_parameters_for_law,
)
from pyvale.vfm.project_definition import (
    IdentificationProject,
    PhaseResult,
    TestData,
)
from pyvale.vfm.project_definition import resolve_parameter_initial_value_scalar
from pyvale.vfm.spatial_parameterisation import build_parameter_state


def build_mechanical_properties_from_project(
    project: IdentificationProject,
) -> MechanicalProperties:
    """Build the resolved material definition used by constitutive updates.

    At this stage each parameter is just a scalar placeholder. During
    identification the active parameter maps are rebuilt from the spatial
    parameterisation state and replace the corresponding placeholders.
    """

    parameters = {}
    for parameter_name in required_parameters_for_law(project.constituitive_law):
        try:
            parameter_definition = project.parameters[parameter_name.name]
        except KeyError as error:
            raise ValueError(
                f"Project is missing parameter '{parameter_name.name}'."
            ) from error

        initial_value = resolve_parameter_initial_value_scalar(parameter_definition)
        if initial_value is None:
            raise ValueError(
                f"Parameter '{parameter_name.name}' needs an initial_value."
            )

        parameters[parameter_name] = KnownParameter(
            value=float(initial_value),
        )

    mechanical_properties = MechanicalProperties(
        constituitive_law=project.constituitive_law,
        parameters=parameters,
    )
    mechanical_properties.validate()
    return mechanical_properties


def run_identification(project: IdentificationProject) -> list[PhaseResult]:
    """Run the phase list defined in the project from start to finish."""

    if project.test_data_path is None:
        raise ValueError("The project does not define a test_data_path.")

    # Load test data from file
    try:
        test_data = load_parsed_test_data(project.test_data_path)
    except Exception as error:
        print(f"Error loading parsed test data: {error}")
        raise
    print(f"Loaded test data from {test_data.source_path} with strain shape {test_data.strain.shape}.")

    # Build base mechanical properties from the project definition. These will be updated
    base_mechanical_properties = build_mechanical_properties_from_project(project)
    print(
        f"Prepared base mechanical properties for "
        f"{project.constituitive_law.name}."
    )

    phase_results: list[PhaseResult] = []
    previous_result: PhaseResult | None = None

    for phase_index, phase_definition in enumerate(project.phases, start=1):
        print(
            f"\nStarting phase {phase_index}/{len(project.phases)}: "
            f"{phase_definition.name}"
        )

        # Assemble dict of parameter states for this phase by building each one from the project definition
        parameter_states = {}
        for parameter_name, parameterisation_specs in phase_definition.parameterisations.items():
            parameter_definition = project.parameters[parameter_name]
            print(
                f"  Building parameter state for {parameter_name} "
                f"with {len(parameterisation_specs)} parameterisation row(s)."
            )
            # Each ParameterState has list of parameterisations 
            parameter_state = build_parameter_state(
                parameter_name=parameter_name,
                parameter_definition=parameter_definition,
                parameterisation_specs=parameterisation_specs,
                previous_result=previous_result,
            )
            source_map = None
            if previous_result is not None:
                source_map = previous_result.parameter_maps.get(parameter_name)
            # 
            parameter_state.initialise_from_map(test_data, source_map)
            parameter_states[parameter_name] = parameter_state
        print(f"  Built {len(parameter_states)} parameter states.")

        # Run identification for this phase
        if project.constituitive_law is ConstituitiveLaw.Elastic:
            phase_result = run_linear_identification(test_data, phase_definition)
        else:
            phase_result = run_nonlinear_identification(
                test_data=test_data,
                phase_definition=phase_definition,
                base_mechanical_properties=base_mechanical_properties,
                parameter_states=parameter_states,
            )

        phase_results.append(phase_result)
        previous_result = phase_result
        print(
            f"Completed {phase_definition.name} with cost "
            f"{phase_result.cost:.6g} and metrics {phase_result.metric_values}."
        )

    return phase_results
