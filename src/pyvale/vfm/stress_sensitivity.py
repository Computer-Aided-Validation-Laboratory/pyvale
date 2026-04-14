from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.mechanical_properties import (
    KnownParameter,
    MechanicalProperties,
    coerce_parameter_name,
)
from pyvale.vfm.project_definition import TestData
from pyvale.vfm.radial_return import radial_return
from pyvale.vfm.spatial_parameterisation import (
    ParameterState,
    ParameterisationDof,
    resolve_parameter_maps,
)


@dataclass(slots=True)
class StressSensitivity:
    """Total and incremental stress sensitivity for one active DOF."""

    total: npt.NDArray[np.float64]
    incremental: npt.NDArray[np.float64]


def calculate_stress_sensitivity(
    stress_reference: npt.NDArray[np.float64],
    test_data: TestData,
    base_mechanical_properties: MechanicalProperties,
    parameter_states: dict[str, ParameterState],
    active_dofs: list[ParameterisationDof],
    perturbation_factor: float = 0.15,
) -> dict[str, StressSensitivity]:
    """Perturb each active DOF once and measure the stress change.

    The parameter-state objects hold the current candidate point in the
    optimisation. For each active DOF we clone that state, apply a small
    perturbation, rebuild the parameter maps, rerun the constitutive update,
    and compare the perturbed stress to the reference stress.
    """

    stress_sensitivities: dict[str, StressSensitivity] = {}
    timestep_deltas = _compute_timestep_deltas(test_data.time)

    for dof in active_dofs:
        perturbed_states = deepcopy(parameter_states)
        _apply_perturbation(perturbed_states, dof.uid, perturbation_factor)

        parameter_maps = resolve_parameter_maps(perturbed_states, test_data)
        resolved_properties = _resolve_mechanical_properties(
            base_mechanical_properties,
            parameter_maps,
        )
        perturbed_stress, _, _, _ = radial_return(test_data.strain, resolved_properties)

        total_stress_sensitivity = stress_reference - perturbed_stress
        incremental_stress_sensitivity = np.zeros_like(total_stress_sensitivity)
        incremental_stress_sensitivity[1:, :, :, :] = np.diff(
            total_stress_sensitivity,
            axis=0,
        )
        incremental_stress_sensitivity = (
            incremental_stress_sensitivity
            / timestep_deltas[:, np.newaxis, np.newaxis, np.newaxis]
        )

        stress_sensitivities[dof.uid] = StressSensitivity(
            total=total_stress_sensitivity,
            incremental=incremental_stress_sensitivity,
        )

    return stress_sensitivities


def _apply_perturbation(
    parameter_states: dict[str, ParameterState],
    dof_uid: str,
    perturbation_factor: float,
) -> None:
    for parameter_state in parameter_states.values():
        for parameterisation in parameter_state.parameterisations:
            for dof in parameterisation.collect_dofs():
                if dof.uid != dof_uid:
                    continue

                dof.value = float(
                    np.clip(
                        dof.value * (1.0 - perturbation_factor),
                        dof.lower_bound,
                        dof.upper_bound,
                    )
                )
                return

    raise ValueError(f"Could not find active DOF '{dof_uid}' for perturbation.")


def _compute_timestep_deltas(time: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if time.size == 0:
        return np.ones(0, dtype=np.float64)

    timestep_deltas = np.ones_like(time, dtype=np.float64)
    if time.size > 1:
        timestep_deltas[1:] = np.diff(time)
        timestep_deltas[0] = timestep_deltas[1]

    timestep_deltas[timestep_deltas == 0.0] = 1.0
    return timestep_deltas


def _resolve_mechanical_properties(
    base_mechanical_properties: MechanicalProperties,
    parameter_maps: dict[str, npt.NDArray[np.float64]],
) -> MechanicalProperties:
    resolved_parameters = dict(base_mechanical_properties.parameters)

    for parameter_name, parameter_map in parameter_maps.items():
        resolved_parameters[coerce_parameter_name(parameter_name)] = KnownParameter(
            parameter_map
        )

    return MechanicalProperties(
        constituitive_law=base_mechanical_properties.constituitive_law,
        parameters=resolved_parameters,
    )
