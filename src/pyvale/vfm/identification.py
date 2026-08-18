import copy
from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np

from pyvale.vfm.constlaw import EIdentificationType, IConstitutiveLaw
from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.identificationconfig import IdentificationConfig
from pyvale.vfm.identificationconfig import IdentificationPhase
from pyvale.vfm.identificationresult import (
    IdentificationHistory,
    IdentificationMetadata,
    IdentificationResult,
    OptimisationOutcome,
    PhaseResult,
    RefinementEvent,
    SolveResult,
    generic_completed_solve_result,
    input_metadata_from_experiment_data,
    snapshot_identification_config,
    snapshot_phase,
    snapshot_phase_config,
    snapshot_refinement_action,
    snapshot_refinement_policy,
    start_run_metadata,
    summarise_refinement_action,
    summarise_refinement_target,
)
from pyvale.vfm.metric import IMetric
from pyvale.vfm.progress import ProgressEvent, emit_progress
from pyvale.vfm.refinement import IRefinementPolicy
from pyvale.vfm.refinement import RefinementContext
from pyvale.vfm.spatialparam import ISpatialParameterisation
from pyvale.vfm.validation import run_validation
from pyvale.vfm.spatialparam import (
    PhaseSpatialState,
    evaluate_parameterisations_to_map,
)


def run_identification(
    experiment_data: ExperimentData,
    identification_config: IdentificationConfig,
    *,
    input_source: str | Path | None = None,
    progress_callback=None,
) -> IdentificationResult:
    """
    Run a VFM identification and return the result.

    Validates the inputs, then executes each configured identification phase in
    order, with the parameters from one phase becoming the initial guess for
    the next.

    Parameters
    ----------
    experiment_data : ExperimentData
        The measured full-field strain, geometry, boundary conditions and
        timesteps
    identification_config : IdentificationConfig
        The constitutive law, initial parameters (with bounds) and the
        identification phases to run
    progress_callback
        Optional callable receiving lightweight progress events.
        If None, no progress events are emitted. If provided, the callable should
        accept a ProgressEvent as an argument and can be used to report progress to the user.

    Returns
    -------
    IdentificationResult
        The final identified parameter map for each constitutive parameter,
        together with an ``IdentificationHistory`` holding one snapshot per
        phase (each capturing the phase's spatial parameterisations and their
        degree-of-freedom values, taken at the end of the phase)
    """
    phase_count = len(identification_config.phases)

    # Emit a progress event to indicate the start of the identification process.
    emit_progress(
        progress_callback,
        ProgressEvent("Starting identification", kind="run_started"),
    )

    # Capture start time and system metadata for log
    run_metadata = start_run_metadata()

    # Validate the experiment data and identification configuration before proceeding.
    run_validation(experiment_data, identification_config)

    # Snapshot the identification configuration for log
    config_snapshot = snapshot_identification_config(
        identification_config
    )

    # Capture input metadata from the experiment data and source path for log
    input_metadata = input_metadata_from_experiment_data(
        experiment_data,
        source_path=input_source,
    )

    # Initialise identification history with empty phases list (populated as phases complete)
    history = IdentificationHistory()

    match identification_config.constitutive_law.get_identification_type():
        # The current implementation assumes either linear or nonlinear identification is
        # being performed (not a mix thoughout). As such, it is recommended to first run a 
        # linear identification to obtain elastic parameters, and then use those as known 
        # parameters in a subsequent nonlinear identification. This decision was made to 
        # simplify implementation and recognises generally different data (e.g. load steps) are used for each
        # TODO: implement linear case
        case EIdentificationType.Linear:
            ...
        case EIdentificationType.Nonlinear:
            # Store the specimen datapoint map size for use in phase runtimes
            parameter_map_size = np.array(
                experiment_data.specimen_geometry.x.shape,
                dtype=np.uint32
            )

            # Iterate through each identification phase, preparing and solving them in order.
            for phase_index, phase in enumerate(identification_config.phases):
                # Emit a progress event to indicate the start of the current phase.
                _emit_phase_progress(
                    progress_callback,
                    kind="phase_started",
                    phase_index=phase_index,
                    phase_count=phase_count,
                    message="started",
                )

                # Prepare the phase runtime, which includes copying spatial parameterisations, metrics,
                # and refinement policy while preserving shared support objects. Metrics and supports are
                #  then prepared based on the experiment data.
                phase_runtime = prepare_phase_runtime(
                    phase,
                    experiment_data,
                )

                # Initialise a PhaseResult to store the results of the current phase, including solve results and refinement events.
                phase_result = PhaseResult(
                    phase_index=phase_index,
                    config=snapshot_phase_config(phase_index, phase),
                )

                solve_iteration = 0
                while True:
                    # Initialise / update DOFs using the current maps (initial or identified)
                    phase_runtime.initialise_from_constitutive_parameters(
                        identification_config.parameters,
                        parameter_map_size,
                    )

                    # Collect the initial DOF values for logging and comparison after optimisation.
                    initial_dofs = _collect_dof_values(
                        phase_runtime.spatial_state.collect_degrees_of_freedom()
                    )

                    # Emit a progress event to indicate the start of the current solve iteration within the phase.
                    _emit_solve_progress(
                        progress_callback,
                        kind="solve_started",
                        phase_index=phase_index,
                        phase_count=phase_count,
                        solve_iteration=solve_iteration,
                        message="started",
                    )

                    solve_started_at = time.perf_counter()
                    # Optimise the active DOFs to minimise the objective.
                    optimisation_result = phase.optimiser.optimise(
                        identification_config.constitutive_law,
                        parameter_map_size,
                        phase_runtime.spatial_state.spatial_parameterisations,
                        phase_runtime.metrics,
                        phase.objective_function,
                        experiment_data,
                        progress_callback=_phase_progress_callback(
                            progress_callback,
                            phase_index=phase_index,
                            phase_count=phase_count,
                            solve_iteration=solve_iteration,
                        ),
                    )
                    solve_runtime = time.perf_counter() - solve_started_at

                    # Unpack the optimisation result into optimised spatial parameterisations and a SolveResult object.
                    optimised_spatial_parameterisations, solve_result = _unpack_optimisation_result(
                        optimisation_result,
                        phase.optimiser,
                        solve_iteration,
                        solve_runtime,
                        initial_dofs,
                    )

                    # Adopt optimiser output and update the global maps.
                    phase_runtime.adopt_spatial_parameterisations(
                        optimised_spatial_parameterisations
                    )

                    # Gather identified DOFs for logging
                    solve_result.final_dofs = _collect_dof_values(
                        phase_runtime.spatial_state.collect_degrees_of_freedom()
                    )

                    # Append solve result to list of solves for this phase 
                    phase_result.solve_results.append(solve_result)

                    # Emit a progress event to indicate the completion of the current solve iteration within the phase
                    _emit_solve_progress(
                        progress_callback,
                        kind="solve_finished",
                        phase_index=phase_index,
                        phase_count=phase_count,
                        solve_iteration=solve_iteration,
                        message="solve finished",
                        evaluation_count=solve_result.num_evaluations,
                        elapsed_seconds=solve_runtime,
                    )

                    # Update the phase parameter maps with the optimised values from this solve iteration.
                    phase_runtime.update_constitutive_parameter_maps(
                        identification_config.parameters,
                        parameter_map_size,
                    )

                    # If no refinement defined for current phase, proceed to next phase
                    if phase_runtime.refinement_policy is None:
                        break

                    # Emit progress event to indicate start of refinement
                    _emit_solve_progress(
                        progress_callback,
                        kind="refinement_started",
                        phase_index=phase_index,
                        phase_count=phase_count,
                        solve_iteration=solve_iteration,
                        message="refinement started",
                    )

                    # Gather refinement context (single object containing experiment data,
                    #  constitutive law, constitutive parameters, parameter map size, and current parameter maps)
                    context = phase_runtime.build_refinement_context(
                        identification_config.constitutive_law,
                        identification_config.parameters,
                        parameter_map_size,
                        experiment_data,
                    )

                    # Check if the refinement policy proposes a refinement action based on the current phase runtime and context.
                    action = phase_runtime.refinement_policy.propose(
                        phase_runtime,
                        context,
                    )

                    # If no refinement action is proposed, break the loop and proceed to the next phase.
                    if action is None:
                        _emit_solve_progress(
                            progress_callback,
                            kind="refinement_finished",
                            phase_index=phase_index,
                            phase_count=phase_count,
                            solve_iteration=solve_iteration,
                            message="no refinement proposed",
                        )
                        break

                    # If refinement action is proposed, summarise the target (to be refined)
                    # before applying the refinement, for logging purposes.
                    target_before = _summarise_refinement_policy_target(
                        phase_runtime,
                        phase_runtime.refinement_policy,
                    )

                    # Apply refinement action to the phase runtime
                    action.apply(phase_runtime, context)

                    # After applying the refinement, prepare the phase runtime again to ensure that any new supports,
                    # parameterisations or metrics are correctly initialised and ready for the next solve iteration.
                    phase_runtime.prepare(experiment_data)

                    # Summarise the refinement target after refinement for logging purposes.
                    target_after = _summarise_refinement_policy_target(
                        phase_runtime,
                        phase_runtime.refinement_policy,
                    )

                    # Record the refinement event in the phase result, 
                    # including the action taken and summaries of the target before and after refinement.
                    phase_result.refinement_events.append(
                        RefinementEvent(
                            event_index=len(phase_result.refinement_events),
                            policy=snapshot_refinement_policy(
                                phase_runtime.refinement_policy
                            ),
                            action=snapshot_refinement_action(action),
                            trigger_summary=summarise_refinement_action(
                                action,
                                before_summary=target_before,
                                after_summary=target_after,
                            ),
                            before_summary=target_before,
                            after_summary=target_after,
                        )
                    )

                    # Emit progress event to indicate refinement has been applied
                    _emit_solve_progress(
                        progress_callback,
                        kind="refinement_finished",
                        phase_index=phase_index,
                        phase_count=phase_count,
                        solve_iteration=solve_iteration,
                        message="refinement applied",
                    )

                    # Increment the solve iteration counter and continue the loop
                    solve_iteration += 1

                # After solve loop, snapshot the phase's final state 
                # (spatial parameterisations and their DOF values)
                phase_result.final_snapshot = (
                    snapshot_phase(
                        phase_runtime.spatial_parameterisations
                    )
                )

                # Append the completed phase result to the overall identification history
                history.phases.append(phase_result)
                _emit_phase_progress(
                    progress_callback,
                    kind="phase_finished",
                    phase_index=phase_index,
                    phase_count=phase_count,
                    message="finished",
                )

                # Continue to next phase, using the updated parameter maps from this phase
                # as the initial guess for the next phase.

    # After all phases are complete, gather the final parameter maps
    parameter_maps = {
        name: np.asarray(parameter.map, dtype=np.float64)
        for name, parameter in identification_config.parameters.items()
    }

    # Calculate the final stress field using the constitutive law and the identified parameter maps.
    final_stress = identification_config.constitutive_law.calculate_stress(
        experiment_data.strain,
        parameter_maps,
    )

    # Capture total runtime in run metadata
    run_metadata.finish()

    # Emit a progress event to indicate the completion of the identification process.
    emit_progress(
        progress_callback,
        ProgressEvent(
            "Identification finished",
            kind="run_finished",
            elapsed_seconds=run_metadata.runtime_seconds,
        ),
    )

    # Return the final identification result, including parameter maps, history, final stress, and metadata.
    return IdentificationResult(
        parameter_maps=parameter_maps,
        history=history,
        final_stress=final_stress,
        metadata=IdentificationMetadata(
            run=run_metadata,
            input=input_metadata,
            config=config_snapshot,
        ),
    )


def _emit_phase_progress(
    progress_callback,
    *,
    kind: str,
    phase_index: int,
    phase_count: int,
    message: str,
) -> None:
    if progress_callback is None:
        return

    emit_progress(
        progress_callback,
        ProgressEvent(
            f"Phase {phase_index + 1}/{phase_count} {message}",
            kind=kind,
        ),
    )


def _emit_solve_progress(
    progress_callback,
    *,
    kind: str,
    phase_index: int,
    phase_count: int,
    solve_iteration: int,
    message: str,
    evaluation_count: int | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    if progress_callback is None:
        return

    full_message = (
        f"Phase {phase_index + 1}/{phase_count}, "
        f"solve {solve_iteration + 1} {message}"
    )
    if evaluation_count is not None:
        full_message += f", evaluations: {evaluation_count}"

    emit_progress(
        progress_callback,
        ProgressEvent(
            full_message,
            kind=kind,
            elapsed_seconds=elapsed_seconds,
        ),
    )


def _phase_progress_callback(
    progress_callback,
    *,
    phase_index: int,
    phase_count: int,
    solve_iteration: int,
):
    if progress_callback is None:
        return None

    def _callback(event: ProgressEvent) -> None:
        message = (
            f"Phase {phase_index + 1}/{phase_count}, "
            f"solve {solve_iteration + 1}, {event.message}"
        )
        emit_progress(
            progress_callback,
            ProgressEvent(
                message,
                kind=event.kind,
                elapsed_seconds=event.elapsed_seconds,
            ),
        )

    return _callback


def _collect_dof_values(
    degrees_of_freedom: list,
) -> list[float]:
    return [float(dof.value) for dof in degrees_of_freedom]


def _unpack_optimisation_result(
    optimisation_result: OptimisationOutcome | dict[str, list[ISpatialParameterisation]],
    optimiser: object,
    solve_iteration: int,
    runtime_seconds: float,
    initial_dofs: list[float],
) -> tuple[dict[str, list[ISpatialParameterisation]], SolveResult]:
    """
    Return the optimised spatial parameterisations and a populated SolveResult.
    """
    if not isinstance(optimisation_result, OptimisationOutcome):
        return (
            optimisation_result,
            generic_completed_solve_result(
                solve_iteration=solve_iteration,
                optimiser=optimiser,
                runtime_seconds=runtime_seconds,
                initial_dofs=initial_dofs,
                final_dofs=[],
            ),
        )

    solve_result = optimisation_result.solve_result or generic_completed_solve_result(
        solve_iteration=solve_iteration,
        optimiser=optimiser,
        runtime_seconds=runtime_seconds,
        initial_dofs=initial_dofs,
        final_dofs=[],
    )

    solve_result.solve_iteration = solve_iteration
    solve_result.runtime_seconds = solve_result.runtime_seconds or float(runtime_seconds)
    solve_result.initial_dofs = solve_result.initial_dofs or list(initial_dofs)

    return optimisation_result.spatial_parameterisations, solve_result


def _summarise_refinement_policy_target(
    phase_runtime: "PhaseRuntime",
    refinement_policy: IRefinementPolicy | None,
) -> dict:
    if refinement_policy is None:
        return {"kind": "none"}

    target = getattr(refinement_policy, "target", None)
    try:
        target = phase_runtime.resolve_support_target(target)
    except Exception as exc:
        return {
            "kind": "unknown",
            "target_type": type(target).__name__,
            "note": f"Could not resolve refinement target: {exc}",
        }
    return summarise_refinement_target(target)


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
        """Initialise / update DOFs using the current maps (initial or identified)."""

        self.spatial_state.initialise_from_constitutive_parameters(
            constitutive_parameters,
            size,
        )

    def adopt_spatial_parameterisations(
        self,
        spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
    ) -> None:
        """Adopt optimiser output while keeping metrics linked to shared supports.
        
        The optimiser returns a list of spatial parameterisations (which may include supports)
        that are different from the original ones. This step updates the phase runtime to use
        the new spatial parameterisations while preserving shared supports and metrics.
        """

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
        """Build solved-state data for phase-level refinement policies.
        
        Gathers experiment data, constitutive law, constitutive parameters, parameter map size,
        and current parameter maps into a context object for use by refinement policies.
        """

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
