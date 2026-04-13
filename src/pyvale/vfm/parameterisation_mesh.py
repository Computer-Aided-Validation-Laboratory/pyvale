from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyvale.vfm.project_definition import ParameterDefinition, ParameterisationSpec, PhaseResult, TestData
from pyvale.vfm.spatial_parameterisation import BaseParameterisation, ParameterisationDof


@dataclass(slots=True)
class MeshParameterisation(BaseParameterisation):
    parameter_name: str
    options: dict[str, Any] = field(default_factory=dict)
    kind: str = "mesh"

    def collect_dofs(self) -> list[ParameterisationDof]:
        return []

    def to_map(self, test_data: TestData):
        raise NotImplementedError(
            f"Mesh parameterisation for '{self.parameter_name}' is scaffolded "
            "but not implemented yet."
        )


def build_mesh_parameterisation(
    parameter_name: str,
    parameter_definition: ParameterDefinition,
    parameterisation_spec: ParameterisationSpec,
    previous_result: PhaseResult | None = None,
) -> BaseParameterisation:
    return MeshParameterisation(
        parameter_name=parameter_name,
        options=parameterisation_spec.options,
    )

