from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pyvale.vfm.mechanical_properties import EParameterName

if TYPE_CHECKING:
    from pyvale.vfm.project_definition import ParameterDefinition, ParameterisationSpec, PhaseResult, TestData


@dataclass(slots=True)
class ParameterisationDof:
    """One optimisation degree of freedom owned by a parameterisation."""

    uid: str
    group: str
    value: float
    lower_bound: float
    upper_bound: float
    active: bool = True


class BaseParameterisation(ABC):
    """Small shared interface for spatial parameterisations."""

    kind: str = "unknown"

    def prepare(self, test_data: TestData) -> None:
        """Prepare reusable data before the optimisation loop."""

    def initialise(
        self,
        test_data: TestData,
        source_map: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """Initialise the parameterisation state from a source map if needed."""

    @abstractmethod
    def collect_dofs(self) -> list[ParameterisationDof]:
        """Return the optimisation DOFs used by this parameterisation."""

    def active_dofs(self) -> list[ParameterisationDof]:
        return [dof for dof in self.collect_dofs() if dof.active]

    def update_from_values(self, values_by_uid: dict[str, float]) -> None:
        for dof in self.collect_dofs():
            if dof.uid in values_by_uid:
                dof.value = values_by_uid[dof.uid]

    @abstractmethod
    def to_map(self, test_data: TestData) -> npt.NDArray[np.float64]:
        """Resolve the parameter contribution onto the full 2D grid."""


@dataclass(slots=True)
class ParameterState:
    """All contributions currently used to build one constitutive parameter."""

    parameter_name: EParameterName
    parameterisations: list[BaseParameterisation] = field(default_factory=list)

    def prepare(self, test_data: TestData) -> None:
        for parameterisation in self.parameterisations:
            parameterisation.prepare(test_data)

    def initialise_from_map(
        self,
        test_data: TestData,
        source_map: npt.NDArray[np.float64] | None,
    ) -> None:
        for parameterisation in self.parameterisations:
            parameterisation.initialise(test_data, source_map)

    def collect_dofs(self) -> list[ParameterisationDof]:
        dofs: list[ParameterisationDof] = []
        for parameterisation in self.parameterisations:
            dofs.extend(parameterisation.collect_dofs())
        return dofs

    def to_map(self, test_data: TestData) -> npt.NDArray[np.float64]:
        parameter_map = np.zeros((test_data.x.shape[0], test_data.x.shape[1]), dtype=np.float64)

        for parameterisation in self.parameterisations:
            parameter_map = parameter_map + parameterisation.to_map(test_data)

        parameter_map[~test_data.specimen_mask] = np.nan
        return parameter_map


def collect_active_dofs(
    parameter_states: dict[str, ParameterState],
) -> list[ParameterisationDof]:
    active_dofs: list[ParameterisationDof] = []

    for parameter_state in parameter_states.values():
        for dof in parameter_state.collect_dofs():
            if dof.active:
                active_dofs.append(dof)

    return active_dofs


def pack_dof_vector(
    active_dofs: list[ParameterisationDof],
) -> tuple[npt.NDArray[np.float64], list[tuple[float, float]]]:
    if not active_dofs:
        return np.zeros(0, dtype=np.float64), []

    initial_values = np.array(
        [dof.value for dof in active_dofs],
        dtype=np.float64,
    )
    bounds = [
        (dof.lower_bound, dof.upper_bound)
        for dof in active_dofs
    ]
    return initial_values, bounds


def update_parameter_states_from_vector(
    parameter_states: dict[str, ParameterState],
    active_dofs: list[ParameterisationDof],
    vector: npt.NDArray[np.float64],
) -> None:
    values_by_uid = {
        dof.uid: float(value)
        for dof, value in zip(active_dofs, vector, strict=True)
    }

    for parameter_state in parameter_states.values():
        for parameterisation in parameter_state.parameterisations:
            parameterisation.update_from_values(values_by_uid)


def resolve_parameter_maps(
    parameter_states: dict[str, ParameterState],
    test_data: TestData,
) -> dict[str, npt.NDArray[np.float64]]:
    return {
        parameter_name: parameter_state.to_map(test_data)
        for parameter_name, parameter_state in parameter_states.items()
    }


def build_parameter_state(
    parameter_name: str,
    parameter_definition: ParameterDefinition,
    parameterisation_specs: list[ParameterisationSpec],
    previous_result: PhaseResult | None = None,
) -> ParameterState:
    """Build the spatial state for a single parameter in one phase."""

    from pyvale.vfm.parameterisation_basis_function import build_basis_function_parameterisation
    from pyvale.vfm.parameterisation_homogeneous import build_homogeneous_parameterisation
    from pyvale.vfm.parameterisation_linked import build_linked_parameterisation
    from pyvale.vfm.parameterisation_mesh import build_mesh_parameterisation
    from pyvale.vfm.parameterisation_slice import build_slice_parameterisation

    builders = {
        "known": build_homogeneous_parameterisation,
        "homogeneous": build_homogeneous_parameterisation,
        "mesh": build_mesh_parameterisation,
        "basis_function": build_basis_function_parameterisation,
        "slice_wise": build_slice_parameterisation,
        "slicewise": build_slice_parameterisation,
        "linked": build_linked_parameterisation,
    }

    parameterisations: list[BaseParameterisation] = []

    for parameterisation_spec in parameterisation_specs:
        try:
            builder = builders[parameterisation_spec.kind]
        except KeyError as error:
            raise ValueError(
                f"Unsupported parameterisation kind '{parameterisation_spec.kind}'."
            ) from error

        parameterisation = builder(
            parameter_name=parameter_name,
            parameter_definition=parameter_definition,
            parameterisation_spec=parameterisation_spec,
            previous_result=previous_result,
        )
        parameterisations.append(parameterisation)

    return ParameterState(
        parameter_name=parameter_definition.name,
        parameterisations=parameterisations,
    )
