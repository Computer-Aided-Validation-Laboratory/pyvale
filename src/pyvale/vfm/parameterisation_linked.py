from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy.typing as npt

from pyvale.vfm.project_definition import ParameterDefinition, ParameterisationSpec, PhaseResult, TestData
from pyvale.vfm.spatial_parameterisation import BaseParameterisation, ParameterisationDof


@dataclass(slots=True)
class LinkedParameterisation(BaseParameterisation):
    """Reuse a previous phase parameterisation and unfreeze selected DOF groups."""

    parameter_name: str
    base_parameterisations: list[BaseParameterisation]
    free_dof_groups: set[str]
    kind: str = "linked"

    def __post_init__(self) -> None:
        for parameterisation in self.base_parameterisations:
            for dof in parameterisation.collect_dofs():
                dof.active = dof.group in self.free_dof_groups

    def prepare(self, test_data: TestData) -> None:
        for parameterisation in self.base_parameterisations:
            parameterisation.prepare(test_data)

    def initialise(
        self,
        test_data: TestData,
        source_map: npt.NDArray[np.float64] | None = None,
    ) -> None:
        for parameterisation in self.base_parameterisations:
            parameterisation.initialise(test_data, source_map)

    def collect_dofs(self) -> list[ParameterisationDof]:
        dofs: list[ParameterisationDof] = []
        for parameterisation in self.base_parameterisations:
            dofs.extend(parameterisation.collect_dofs())
        return dofs

    def update_from_values(self, values_by_uid: dict[str, float]) -> None:
        for parameterisation in self.base_parameterisations:
            parameterisation.update_from_values(values_by_uid)

    def to_map(self, test_data: TestData):
        contribution = None
        for parameterisation in self.base_parameterisations:
            parameter_map = parameterisation.to_map(test_data)
            if contribution is None:
                contribution = parameter_map
            else:
                contribution = contribution + parameter_map

        if contribution is None:
            raise ValueError(
                f"Linked parameter '{self.parameter_name}' has no base parameterisations."
            )

        return contribution


def build_linked_parameterisation(
    parameter_name: str,
    parameter_definition: ParameterDefinition,
    parameterisation_spec: ParameterisationSpec,
    previous_result: PhaseResult | None = None,
) -> BaseParameterisation:
    if previous_result is None:
        raise ValueError(
            f"Linked parameter '{parameter_name}' requires a previous phase result."
        )

    source_parameter = parameterisation_spec.source_parameter or parameter_name
    try:
        source_state = previous_result.parameter_states[source_parameter]
    except KeyError as error:
        raise ValueError(
            f"Linked parameter '{parameter_name}' could not find source "
            f"parameter '{source_parameter}' in the previous phase."
        ) from error

    return LinkedParameterisation(
        parameter_name=parameter_name,
        base_parameterisations=deepcopy(source_state.parameterisations),
        free_dof_groups=set(parameterisation_spec.free_dof_groups),
    )
