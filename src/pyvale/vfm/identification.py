import copy
from dataclasses import dataclass, field

import numpy as np

from pyvale.vfm.constlaw import EIdentificationType, IConstitutiveLaw
from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.identificationconfig import IdentificationConfig
from pyvale.vfm.identificationconfig import IdentificationPhase
from pyvale.vfm.metric import IMetric
from pyvale.vfm.metricsliceforce import SliceWiseForceReconstructionMetric
from pyvale.vfm.spatialparam import ISpatialParameterisation
from pyvale.vfm.spatialparamslicewise import SupportSlice
from pyvale.vfm.validation import (
    validate_experiment_data,
    validate_identification_config,
)
from pyvale.vfm.spatialparam import (
    PhaseSpatialState,
    evaluate_parameterisations_to_map,
)


@dataclass(slots=True)
class PhaseRuntime:
    """Prepared runtime state for one identification phase.

    The phase definition is a declarative configuration. This runtime object
    owns the working copies used during solving, including any prepared shared
    supports and prepared metric state.
    """

    spatial_parameterisations: dict[str, list[ISpatialParameterisation]]
    metrics: list[IMetric]
    spatial_state: PhaseSpatialState = field(init=False)

    def __post_init__(self) -> None:
        self.rebuild_spatial_state()

    def rebuild_spatial_state(self) -> None:
        """Rebuild the DOF-routing view after support or parameter changes."""

        self.spatial_state = PhaseSpatialState(self.spatial_parameterisations)

    def prepare(
        self,
        experiment_data: ExperimentData,
    ) -> None:
        """Prepare shared supports and metrics for the current runtime state."""

        # Update the spatial state to reflect any changes to the spatial parameterisations
        self.rebuild_spatial_state()
        # Prepare any shared supports and metrics for the current runtime state
        self.spatial_state.prepare(experiment_data)
        for metric in self.metrics:
            metric.initialise(experiment_data)

    def initialise_from_constitutive_parameters(
        self,
        constitutive_parameters: dict[str, ConstitutiveParameter],
        size: np.ndarray,
    ) -> None:
        """Project the current constitutive maps onto the active runtime DOFs."""

        self.spatial_state.initialise_from_constitutive_parameters(
            constitutive_parameters,
            size,
        )

    def adopt_spatial_parameterisations(
        self,
        spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
    ) -> None:
        """Adopt optimiser output while keeping metrics linked to shared supports."""

        support_replacements = _map_updated_supports(
            self.spatial_parameterisations,
            spatial_parameterisations,
        )
        for metric in self.metrics:
            support = getattr(metric, "support", None)
            replacement = support_replacements.get(id(support))
            if replacement is None:
                continue

            set_support = getattr(metric, "set_support", None)
            if set_support is not None:
                set_support(replacement)
            else:
                metric.support = replacement

        self.spatial_parameterisations = spatial_parameterisations
        self.rebuild_spatial_state()


def _map_updated_supports(
    old_spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
    new_spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
) -> dict[int, object]:
    """Map old support object ids to optimiser-updated support objects."""

    support_replacements: dict[int, object] = {}
    for param_name, old_parameterisation_list in old_spatial_parameterisations.items():
        new_parameterisation_list = new_spatial_parameterisations[param_name]
        for old_sp, new_sp in zip(
            old_parameterisation_list,
            new_parameterisation_list,
            strict=True,
        ):
            old_support = getattr(old_sp, "support", None)
            new_support = getattr(new_sp, "support", None)
            if old_support is None or new_support is None:
                continue
            support_replacements[id(old_support)] = new_support

    return support_replacements


def _collect_unique_supports(
    phase_runtime: PhaseRuntime,
) -> list[object]:
    supports: list[object] = []
    support_ids: set[int] = set()

    for support in phase_runtime.spatial_state.supports:
        supports.append(support)
        support_ids.add(id(support))

    for metric in phase_runtime.metrics:
        support = getattr(metric, "support", None)
        if support is None or id(support) in support_ids:
            continue
        supports.append(support)
        support_ids.add(id(support))

    return supports


def _has_refinable_support(
    phase_runtime: PhaseRuntime,
) -> bool:
    return any(
        getattr(support, "refine", False)
        for support in _collect_unique_supports(phase_runtime)
    )


def _update_constitutive_parameter_maps(
    constitutive_parameters: dict[str, ConstitutiveParameter],
    spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
    parameter_map_size: np.ndarray,
) -> None:
    """Write the current runtime parameter maps back to the identification state."""

    for param_name, parameterisation_list in spatial_parameterisations.items():
        constitutive_parameters[param_name].map = evaluate_parameterisations_to_map(
            parameterisation_list,
            parameter_map_size,
        )


def _build_support_slice_refinement_inputs(
    phase_runtime: PhaseRuntime,
    constitutive_law: IConstitutiveLaw,
    constitutive_parameters: dict[str, ConstitutiveParameter],
    experiment_data: ExperimentData,
) -> tuple[
    dict[str, np.ndarray],
    dict[int, np.ndarray],
]:
    """Compute fresh post-solve maps, stress, and slice-force diagnostics."""

    parameter_maps = {
        param_name: np.asarray(parameter.map, dtype=np.float64)
        for param_name, parameter in constitutive_parameters.items()
    }
    stress = constitutive_law.calculate_stress(
        experiment_data.strain,
        parameter_maps,
    )

    force_error_ratio_by_support_id: dict[int, np.ndarray] = {}
    for metric in phase_runtime.metrics:
        if not isinstance(metric, SliceWiseForceReconstructionMetric):
            continue
        result = metric.evaluate_force_recon_error(
            stress,
            experiment_data,
        )
        force_error_ratio_by_support_id[id(metric.support)] = (
            result.weighted_temporal_rms
        )

    return parameter_maps, force_error_ratio_by_support_id


def _perform_optional_refinement(
    phase_runtime: PhaseRuntime,
    experiment_data: ExperimentData,
    *,
    parameter_maps: dict[str, np.ndarray],
    force_error_ratio_by_support_id: dict[int, np.ndarray],
) -> bool:
    """Refine the active runtime state and rebuild prepared supports if needed.

    Shared supports are refined once before checking parameterisation-local
    refinement. Support objects should be mutated in place so existing metric
    and parameterisation references remain valid.
    """

    for support in _collect_unique_supports(phase_runtime):
        if isinstance(support, SupportSlice):
            force_error_ratio = force_error_ratio_by_support_id.get(id(support))
            if not support.should_perform_refinement(
                parameter_maps=parameter_maps,
                spatial_parameterisations=phase_runtime.spatial_parameterisations,
                force_error_ratio=force_error_ratio,
            ):
                continue
            support.perform_refinement(
                parameter_maps=parameter_maps,
                spatial_parameterisations=phase_runtime.spatial_parameterisations,
                force_error_ratio=force_error_ratio,
            )
            phase_runtime.prepare(experiment_data)
            return True

        should_refine = getattr(support, "should_perform_refinement", None)
        perform_refinement = getattr(support, "perform_refinement", None)
        if should_refine is None or perform_refinement is None:
            continue
        if not should_refine():
            continue
        perform_refinement()
        phase_runtime.prepare(experiment_data)
        return True

    refined = False
    for spatial_parameterisation_list in phase_runtime.spatial_parameterisations.values():
        for spatial_parameterisation in spatial_parameterisation_list:
            if not spatial_parameterisation.should_perform_refinement():
                continue
            spatial_parameterisation.perform_refinement()
            refined = True

    if refined:
        phase_runtime.prepare(experiment_data)

    return refined


def prepare_phase_runtime(
    phase: IdentificationPhase,
    experiment_data: ExperimentData,
) -> PhaseRuntime:
    """Prepare one phase runtime once experiment data are available.

    Validation has already checked that the phase configuration is legal.
    This step builds runtime working copies while preserving any support
    sharing declared by the caller, then prepares data-dependent support
    and metric state.
    """

    # Copy spatial parameterisations and metrics together so any shared support
    # objects declared by the caller remain shared in the runtime copies.
    runtime_spatial_parameterisations, runtime_metrics = copy.deepcopy(
        (phase.spatial_parameterisations, phase.metrics)
    )
    phase_runtime = PhaseRuntime(
        spatial_parameterisations=runtime_spatial_parameterisations,
        metrics=runtime_metrics,
    )
    phase_runtime.prepare(experiment_data)
    return phase_runtime


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
                phase_runtime = prepare_phase_runtime(
                    phase,
                    experiment_data,
                )

                while True:
                    # Re-project the current constitutive maps onto the active
                    # runtime parameterisation DOFs before each solve.
                    phase_runtime.initialise_from_constitutive_parameters(
                        identification_config.parameters,
                        parameter_map_size,
                    )

                    optimised_spatial_parameterisations = phase.optimiser.optimise(
                        identification_config.constitutive_law,
                        parameter_map_size,
                        phase_runtime.spatial_state.spatial_parameterisations,
                        phase_runtime.metrics,
                        phase.objective_function,
                        experiment_data
                    )

                    phase_runtime.adopt_spatial_parameterisations(
                        optimised_spatial_parameterisations
                    )
                    _update_constitutive_parameter_maps(
                        identification_config.parameters,
                        phase_runtime.spatial_parameterisations,
                        parameter_map_size,
                    )
                    if _has_refinable_support(phase_runtime):
                        (
                            refinement_parameter_maps,
                            force_error_ratio_by_support_id,
                        ) = _build_support_slice_refinement_inputs(
                            phase_runtime,
                            identification_config.constitutive_law,
                            identification_config.parameters,
                            experiment_data,
                        )
                    else:
                        refinement_parameter_maps = {}
                        force_error_ratio_by_support_id = {}

                    if not _perform_optional_refinement(
                        phase_runtime,
                        experiment_data,
                        parameter_maps=refinement_parameter_maps,
                        force_error_ratio_by_support_id=force_error_ratio_by_support_id,
                    ):
                        break

    return identification_config.parameters
