from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pyvale.vfm.mechanical_properties import EConstituitiveLaw, EParameterName
from pyvale.vfm.project_definition import (
    IdentificationProject,
    MetricSpec,
    OptimiserSpec,
    ParameterDefinition,
    ParameterisationSpec,
    PhaseDefinition,
    create_default_project,
)


def load_project(project_path: str | Path) -> IdentificationProject:
    project_file = Path(project_path)

    with project_file.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    law = EConstituitiveLaw[data.get("constituitive_law", "LinearHardening")]
    project = create_default_project(law)
    project.project_path = project_file
    project.name = data.get("name", project.name)
    project.use_gui = bool(data.get("use_gui", project.use_gui))
    project.notes = data.get("notes", "")

    test_data_path = data.get("test_data_path")
    if test_data_path:
        project.test_data_path = Path(test_data_path)

    parameter_data = data.get("parameters", {})
    for parameter_name, parameter_dict in parameter_data.items():
        coerced_name = EParameterName[parameter_name]
        project.parameters[coerced_name.name] = _load_parameter_definition(
            coerced_name.name,
            parameter_dict,
        )

    phases = []
    for phase_dict in data.get("phases", []):
        # Instantiate a PhaseDefinition from the phase_dict and append it to the phases list
        phase = _load_phase_definition(phase_dict)
        phases.append(phase)
    project.phases = phases


    return project


def save_project(
    project: IdentificationProject,
    project_path: str | Path | None = None,
) -> Path:
    target_path = Path(project_path) if project_path is not None else project.project_path
    if target_path is None:
        raise ValueError("No output path was supplied for saving the project.")

    with target_path.open("w", encoding="utf-8") as handle:
        handle.write(project_to_yaml_text(project))

    return target_path


def project_to_dict(project: IdentificationProject) -> dict[str, Any]:
    return {
        "name": project.name,
        "use_gui": project.use_gui,
        "test_data_path": (
            str(project.test_data_path) if project.test_data_path is not None else None
        ),
        "notes": project.notes,
        "constituitive_law": project.constituitive_law.name,
        "parameters": {
            parameter_name: _parameter_to_dict(parameter)
            for parameter_name, parameter in project.parameters.items()
        },
        "phases": [_phase_to_dict(phase) for phase in project.phases],
    }


def project_to_yaml_text(project: IdentificationProject) -> str:
    """Render the project YAML with a few blank lines for readability."""

    text = yaml.safe_dump(project_to_dict(project), sort_keys=False)
    lines = text.splitlines()
    formatted_lines: list[str] = []

    for line in lines:
        if line.startswith(("constituitive_law:", "phases:")) and formatted_lines:
            if formatted_lines[-1] != "":
                formatted_lines.append("")
        formatted_lines.append(line)

    return "\n".join(formatted_lines) + "\n"


def _load_parameter_definition(
    parameter_name: str,
    data: dict[str, Any],
) -> ParameterDefinition:
    return ParameterDefinition(
        name=EParameterName[parameter_name],
        initial_value_type=str(data.get("initial_value_type", "float")),
        initial_value=data.get("initial_value"),
        lower_bound=data.get("lower_bound"),
        upper_bound=data.get("upper_bound"),
        description=data.get("description", ""),
    )


def _load_phase_definition(data: dict[str, Any]) -> PhaseDefinition:
    parameterisations: dict[str, list[ParameterisationSpec]] = {}

    for parameter_name, parameterisation_list in data.get("parameterisations", {}).items():
        parameterisations[parameter_name] = [
            ParameterisationSpec(
                kind=parameterisation["kind"],
                name=parameterisation.get("name"),
                options=parameterisation.get("options", {}),
                source_phase=parameterisation.get("source_phase"),
                source_parameter=parameterisation.get("source_parameter"),
                free_dof_groups=parameterisation.get("free_dof_groups", []),
            )
            for parameterisation in parameterisation_list
        ]

    metrics = [
        MetricSpec(
            kind=metric["kind"],
            weight=float(metric.get("weight", 1.0)),
            options=metric.get("options", {}),
        )
        for metric in data.get("metrics", [])
    ]

    optimiser_dict = data.get("optimiser", {})
    optimiser = OptimiserSpec(
        kind=optimiser_dict.get("kind", "least_squares"),
        options=optimiser_dict.get("options", {}),
    )

    return PhaseDefinition(
        name=data["name"],
        parameterisations=parameterisations,
        metrics=metrics,
        optimiser=optimiser,
        notes=data.get("notes", ""),
    )


def _phase_to_dict(phase: PhaseDefinition) -> dict[str, Any]:
    return {
        "name": phase.name,
        "notes": phase.notes,
        "optimiser": {
            "kind": phase.optimiser.kind,
            "options": phase.optimiser.options,
        },
        "metrics": [
            {
                "kind": metric.kind,
                "weight": metric.weight,
                "options": metric.options,
            }
            for metric in phase.metrics
        ],
        "parameterisations": {
            parameter_name: [
                {
                    "kind": parameterisation.kind,
                    "name": parameterisation.name,
                    "options": parameterisation.options,
                    "source_phase": parameterisation.source_phase,
                    "source_parameter": parameterisation.source_parameter,
                    "free_dof_groups": parameterisation.free_dof_groups,
                }
                for parameterisation in parameterisations
            ]
            for parameter_name, parameterisations in phase.parameterisations.items()
        },
    }


def _parameter_to_dict(parameter: ParameterDefinition) -> dict[str, Any]:
    parameter_dict: dict[str, Any] = {
        "initial_value_type": parameter.initial_value_type,
        "initial_value": parameter.initial_value,
        "lower_bound": parameter.lower_bound,
        "upper_bound": parameter.upper_bound,
    }
    if parameter.description:
        parameter_dict["description"] = parameter.description
    return parameter_dict
