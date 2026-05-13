from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from pyvale.vfm.mechanical_properties import (
    EParameterName,
    KnownParameter,
    MechanicalProperties,
)
from pyvale.vfm.project_definition import TestData
from pyvale.vfm.radial_return import radial_return
from pyvale.vfm.spatial_parameterisation import (
    ParameterState,
    ParameterisationDof,
    resolve_parameter_maps,
)


StressSensitivityPerturbationType = Literal["constitutive_parameter", "dof"]


@dataclass(slots=True)
class StressSensitivity:
    """Total and incremental stress sensitivity for one perturbation target.

    Both arrays use the constitutive-update layout
    ``(timesteps, components, y, x)``.
    """

    total: npt.NDArray[np.float64]
    incremental: npt.NDArray[np.float64]


def _coerce_perturbation_type(
    perturbation_type: str,
) -> StressSensitivityPerturbationType:
    if perturbation_type in {"constitutive_parameter", "dof"}:
        return perturbation_type

    raise ValueError(
        "perturbation_type must be 'constitutive_parameter' or 'dof', "
        f"got '{perturbation_type}'."
    )


def _compute_timestep_deltas(time: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if time.size == 0:
        return np.ones(0, dtype=np.float64)

    timestep_deltas = np.ones_like(time, dtype=np.float64)
    if time.size > 1:
        timestep_deltas[1:] = np.diff(time)
        timestep_deltas[0] = timestep_deltas[1]

    timestep_deltas[timestep_deltas == 0.0] = 1.0
    return timestep_deltas


def _build_stress_sensitivity(
    stress_reference: npt.NDArray[np.float64],
    perturbed_stress: npt.NDArray[np.float64],
    timestep_deltas: npt.NDArray[np.float64],
) -> StressSensitivity:
    # Match the paper's definition: sensitivity is the reference stress
    # minus the stress reconstructed after perturbing one target.
    total_stress_sensitivity = stress_reference - perturbed_stress

    # The incremental map is a time-difference of the total sensitivity.
    # The first step is kept at zero because there is no previous step.
    incremental_stress_sensitivity = np.zeros_like(total_stress_sensitivity)
    incremental_stress_sensitivity[1:, :, :, :] = np.diff(
        total_stress_sensitivity,
        axis=0,
    )
    incremental_stress_sensitivity = (
        incremental_stress_sensitivity
        / timestep_deltas[:, np.newaxis, np.newaxis, np.newaxis]
    )

    return StressSensitivity(
        total=total_stress_sensitivity,
        incremental=incremental_stress_sensitivity,
    )


def _resolve_mechanical_properties(
    base_mechanical_properties: MechanicalProperties,
    parameter_maps: dict[str, npt.NDArray[np.float64]],
) -> MechanicalProperties:
    resolved_parameters = dict(base_mechanical_properties.parameters)

    for parameter_name, parameter_map in parameter_maps.items():
        resolved_parameters[EParameterName[parameter_name]] = KnownParameter(
            parameter_map
        )

    return MechanicalProperties(
        constituitive_law=base_mechanical_properties.constituitive_law,
        parameters=resolved_parameters,
    )


def _compute_perturbed_stress(
    test_data: TestData,
    base_mechanical_properties: MechanicalProperties,
    parameter_maps: dict[str, npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    resolved_properties = _resolve_mechanical_properties(
        base_mechanical_properties,
        parameter_maps,
    )
    perturbed_stress, _, _, _ = radial_return(test_data.strain, resolved_properties)
    return perturbed_stress


def _collect_active_parameter_names(
    parameter_states: dict[str, ParameterState],
    active_dofs: list[ParameterisationDof],
) -> list[str]:
    active_dof_uids = {dof.uid for dof in active_dofs}
    active_parameter_names: list[str] = []

    for parameter_name, parameter_state in parameter_states.items():
        if any(dof.uid in active_dof_uids for dof in parameter_state.collect_dofs()):
            active_parameter_names.append(parameter_name)

    return active_parameter_names


def _copy_parameter_maps(
    parameter_maps: dict[str, npt.NDArray[np.float64]],
) -> dict[str, npt.NDArray[np.float64]]:
    return {
        parameter_name: parameter_map.copy()
        for parameter_name, parameter_map in parameter_maps.items()
    }


def _perturb_parameter_map(
    parameter_map: npt.NDArray[np.float64],
    specimen_mask: npt.NDArray[np.bool_],
    perturbation_factor: float,
) -> npt.NDArray[np.float64]:
    perturbed_parameter_map = parameter_map.copy()

    # Only perturb real specimen points. Invalid / out-of-specimen entries
    # stay untouched so the constitutive update sees the same masking.
    valid_mask = specimen_mask & np.isfinite(perturbed_parameter_map)
    perturbed_parameter_map[valid_mask] = (
        perturbed_parameter_map[valid_mask] * (1.0 - perturbation_factor)
    )
    return perturbed_parameter_map


def _perturb_single_dof(
    parameter_states: dict[str, ParameterState],
    dof_uid: str,
    perturbation_factor: float,
) -> None:
    for parameter_state in parameter_states.values():
        for parameterisation in parameter_state.parameterisations:
            for dof in parameterisation.collect_dofs():
                if dof.uid != dof_uid:
                    continue

                # The Python parameterisations currently store the active DOF
                # values directly, so the DOF mode applies a multiplicative
                # perturbation to that raw value and clips to its bounds.
                # TODO: dof normalisation would be useful to make the perturbation factor more consistent across DOFs.
                # SEE ROBS NOTES ON THIS
                dof.value = float(
                    np.clip(
                        dof.value * (1.0 - perturbation_factor),
                        dof.lower_bound,
                        dof.upper_bound,
                    )
                )
                return

    raise ValueError(f"Could not find active DOF '{dof_uid}' for perturbation.")


def _compute_parameter_stress_sensitivities(
    stress_reference: npt.NDArray[np.float64],
    test_data: TestData,
    base_mechanical_properties: MechanicalProperties,
    parameter_states: dict[str, ParameterState],
    active_dofs: list[ParameterisationDof],
    perturbation_factor: float,
    timestep_deltas: npt.NDArray[np.float64],
) -> dict[str, StressSensitivity]:
    stress_sensitivities: dict[str, StressSensitivity] = {}
    active_parameter_names = _collect_active_parameter_names(
        parameter_states,
        active_dofs,
    )
    parameter_maps = resolve_parameter_maps(parameter_states, test_data)

    for parameter_name in active_parameter_names:
        perturbed_parameter_maps = _copy_parameter_maps(parameter_maps)
        perturbed_parameter_maps[parameter_name] = _perturb_parameter_map(
            parameter_maps[parameter_name],
            test_data.specimen_mask,
            perturbation_factor,
        )
        perturbed_stress = _compute_perturbed_stress(
            test_data,
            base_mechanical_properties,
            perturbed_parameter_maps,
        )
        stress_sensitivities[parameter_name] = _build_stress_sensitivity(
            stress_reference,
            perturbed_stress,
            timestep_deltas,
        )

    return stress_sensitivities


def _compute_dof_stress_sensitivities(
    stress_reference: npt.NDArray[np.float64],
    test_data: TestData,
    base_mechanical_properties: MechanicalProperties,
    parameter_states: dict[str, ParameterState],
    active_dofs: list[ParameterisationDof],
    perturbation_factor: float,
    timestep_deltas: npt.NDArray[np.float64],
) -> dict[str, StressSensitivity]:
    stress_sensitivities: dict[str, StressSensitivity] = {}

    for dof in active_dofs:
        perturbed_states = deepcopy(parameter_states)
        _perturb_single_dof(perturbed_states, dof.uid, perturbation_factor)

        perturbed_parameter_maps = resolve_parameter_maps(perturbed_states, test_data)
        perturbed_stress = _compute_perturbed_stress(
            test_data,
            base_mechanical_properties,
            perturbed_parameter_maps,
        )
        stress_sensitivities[dof.uid] = _build_stress_sensitivity(
            stress_reference,
            perturbed_stress,
            timestep_deltas,
        )

    return stress_sensitivities


def calculate_stress_sensitivity(
    stress_reference: npt.NDArray[np.float64],
    test_data: TestData,
    base_mechanical_properties: MechanicalProperties,
    parameter_states: dict[str, ParameterState],
    active_dofs: list[ParameterisationDof],
    perturbation_factor: float = 0.15,
    perturbation_type: StressSensitivityPerturbationType = "constitutive_parameter",
) -> dict[str, StressSensitivity]:
    """Compute stress sensitivities used to build sensitivity-based virtual fields.

    The returned dictionary contains one spatiotemporal stress-sensitivity
    history per perturbation target, where each ``StressSensitivity.total``
    and ``StressSensitivity.incremental`` array has shape
    ``(timesteps, components, y, x)``. The downstream SBVF construction then
    builds one virtual field from each of these histories, with the same
    spatiotemporal layout in its virtual-strain field.

    The chosen ``perturbation_type`` determines how many downstream virtual
    fields (VF) will be created by the SBVF metric:

    - ``"constitutive_parameter"``: one sensitivity history per active
      constitutive parameter, so downstream ``nVF = nParameters``.
    - ``"dof"``: one sensitivity history per active optimisation DOF, so
      downstream ``nVF = nDof``.

    The SBVF metric later converts each sensitivity history into one virtual
    field and evaluates one residual per timestep for each VF. That means the
    total residual vector assembled by the metric has length
    ``nVF * nStep``.

    Notes
    -----
    "constitutive_parameter" mode is the paper-aligned option from
    Marek et al. (2017): perturb one constitutive parameter map at a time and
    compare the resulting stress history against the reference stress history.

    "dof" mode is a useful extension for the current optimisation
    framework, but it changes the weighting structure of the metric because
    constitutive parameters with more active DOFs generate more virtual
    fields and therefore more residual blocks. If DOF mode is used, an
    additional weighting strategy should be considered at the metric level.
    """

    resolved_perturbation_type = _coerce_perturbation_type(perturbation_type)
    timestep_deltas = _compute_timestep_deltas(test_data.time)

    if resolved_perturbation_type == "constitutive_parameter":
        return _compute_parameter_stress_sensitivities(
            stress_reference=stress_reference,
            test_data=test_data,
            base_mechanical_properties=base_mechanical_properties,
            parameter_states=parameter_states,
            active_dofs=active_dofs,
            perturbation_factor=perturbation_factor,
            timestep_deltas=timestep_deltas,
        )

    return _compute_dof_stress_sensitivities(
        stress_reference=stress_reference,
        test_data=test_data,
        base_mechanical_properties=base_mechanical_properties,
        parameter_states=parameter_states,
        active_dofs=active_dofs,
        perturbation_factor=perturbation_factor,
        timestep_deltas=timestep_deltas,
    )
