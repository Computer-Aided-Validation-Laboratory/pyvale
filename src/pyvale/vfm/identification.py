import copy
from dataclasses import dataclass, field

import numpy as np

from pyvale.vfm.constlaw import EIdentificationType, IConstitutiveLaw
from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.identificationconfig import IdentificationConfig
from pyvale.vfm.identificationconfig import IdentificationPhase
from pyvale.vfm.metric import IMetric
from pyvale.vfm.refinement import IRefinementPolicy
from pyvale.vfm.refinement import RefinementContext
from pyvale.vfm.spatialparam import ISpatialParameterisation
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
    refinement_policy: IRefinementPolicy | None = None
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

        if self.refinement_policy is not None:
            target = getattr(self.refinement_policy, "target", None)
            replacement = support_replacements.get(id(target))
            if replacement is not None:
                self.refinement_policy.target = replacement

        self.spatial_parameterisations = spatial_parameterisations
        self.rebuild_spatial_state()

    def update_constitutive_parameter_maps(
        self,
        constitutive_parameters: dict[str, ConstitutiveParameter],
        parameter_map_size: np.ndarray,
    ) -> None:
        """Write this phase's parameter maps back to the identification state."""

        for param_name, parameterisation_list in self.spatial_parameterisations.items():
            constitutive_parameters[param_name].map = evaluate_parameterisations_to_map(
                parameterisation_list,
                parameter_map_size,
            )

    def build_refinement_context(
        self,
        constitutive_law: IConstitutiveLaw,
        constitutive_parameters: dict[str, ConstitutiveParameter],
        parameter_map_size: np.ndarray,
        experiment_data: ExperimentData,
    ) -> RefinementContext:
        """Build solved-state data for phase-level refinement policies."""

        return RefinementContext(
            experiment_data=experiment_data,
            constitutive_law=constitutive_law,
            constitutive_parameters=constitutive_parameters,
            parameter_map_size=parameter_map_size,
            parameter_maps={
                param_name: np.asarray(parameter.map, dtype=np.float64)
                for param_name, parameter in constitutive_parameters.items()
            },
        )

    def collect_unique_supports(self) -> list[object]:
        supports: list[object] = []
        support_ids: set[int] = set()

        for support in self.spatial_state.supports:
            supports.append(support)
            support_ids.add(id(support))

        for metric in self.metrics:
            support = getattr(metric, "support", None)
            if support is None or id(support) in support_ids:
                continue
            supports.append(support)
            support_ids.add(id(support))

        return supports

    def get_parameterisation(
        self,
        parameter_name: str,
        index: int,
    ) -> tuple[str, ISpatialParameterisation]:
        return (
            parameter_name,
            self.spatial_parameterisations[parameter_name][index],
        )

    def get_parameterisations_using_support(
        self,
        support: object,
    ) -> list[tuple[str, ISpatialParameterisation]]:
        parameterisations: list[tuple[str, ISpatialParameterisation]] = []
        for parameter_name, parameterisation_list in self.spatial_parameterisations.items():
            for parameterisation in parameterisation_list:
                if getattr(parameterisation, "support", None) is support:
                    parameterisations.append((parameter_name, parameterisation))
        return parameterisations

    def resolve_support_target(
        self,
        target: object,
    ) -> object:
        if isinstance(target, tuple):
            _, parameterisation = self.get_parameterisation(
                target[0],
                target[1],
            )
            return getattr(parameterisation, "support", parameterisation)

        if isinstance(target, str):
            for support in self.collect_unique_supports():
                if getattr(support, "name", None) == target:
                    return support
            raise ValueError(f"No support named '{target}' was found.")

        return target


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

    # Copy spatial parameterisations, metrics, and policy together so shared
    # support objects and policy object-targets remain shared in runtime copies.
    (
        runtime_spatial_parameterisations,
        runtime_metrics,
        runtime_refinement_policy,
    ) = copy.deepcopy(
        (phase.spatial_parameterisations, phase.metrics, phase.refinement_policy)
    )
    phase_runtime = PhaseRuntime(
        spatial_parameterisations=runtime_spatial_parameterisations,
        metrics=runtime_metrics,
        refinement_policy=runtime_refinement_policy,
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
                    # Project the current maps onto the active phase DOFs.
                    phase_runtime.initialise_from_constitutive_parameters(
                        identification_config.parameters,
                        parameter_map_size,
                    )

                    # Optimise the active DOFs to minimise the objective.
                    optimised_spatial_parameterisations = phase.optimiser.optimise(
                        identification_config.constitutive_law,
                        parameter_map_size,
                        phase_runtime.spatial_state.spatial_parameterisations,
                        phase_runtime.metrics,
                        phase.objective_function,
                        experiment_data
                    )

                    # Adopt optimiser output and update the global maps.
                    phase_runtime.adopt_spatial_parameterisations(
                        optimised_spatial_parameterisations
                    )
                    phase_runtime.update_constitutive_parameter_maps(
                        identification_config.parameters,
                        parameter_map_size,
                    )

                    if phase_runtime.refinement_policy is None:
                        break

                    context = phase_runtime.build_refinement_context(
                        identification_config.constitutive_law,
                        identification_config.parameters,
                        parameter_map_size,
                        experiment_data,
                    )
                    action = phase_runtime.refinement_policy.propose(
                        phase_runtime,
                        context,
                    )
                    if action is None:
                        break
                    action.apply(phase_runtime, context)
                    phase_runtime.prepare(experiment_data)

    return identification_config.parameters
