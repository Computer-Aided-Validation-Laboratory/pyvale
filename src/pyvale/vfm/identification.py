import copy
from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import uniform_filter

from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.equilibriumgapaggregation import combine_equilibrium_gap_maps
from pyvale.vfm.metricequilibriumgap import EquilibriumGapMetric
from pyvale.vfm.metricsliceforce import SliceWiseForceReconstructionMetric
from pyvale.vfm.spatialparambasisfuncs import (
    BasisFunctionKernel,
    SpatialParameterisationBasisFunction,
)
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
from pyvale.vfm.metric import IMetric, MetricResult
from pyvale.vfm.objectivefunc import IObjectiveFunction
from pyvale.vfm.objectivefunccombinedfreegi import (
    CombinedForceAndEquilibriumGapObjective,
    CombinedObjectiveBaselineMode,
)
from pyvale.vfm.optimiser import IOptimiser, evaluate_metrics
from pyvale.vfm.progress import ProgressEvent, emit_progress
from pyvale.vfm.refinement import (
    EquilibriumGapBasisGrowthRefinement,
    IRefinementAction,
    IRefinementPolicy,
)
from pyvale.vfm.refinement import RefinementContext
from pyvale.vfm.spatialparam import ISpatialParameterisation
from pyvale.vfm.validation import run_validation
from pyvale.vfm.spatialparam import (
    PhaseSpatialState,
    evaluate_parameterisations_to_map,
)
from pyvale.vfm.spatialparamknown import SpatialParameterisationKnown


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
    input_source : str | Path | None, optional
        The source path of the input data, used for logging.
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
    completed_phase_maps: dict[int, dict[str, np.ndarray]] = {}

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

                # Prepare the phase runtime. This is the mutable,working copy of the phase configuration
                # which includes copying spatial parameterisations, metrics,
                # and refinement policy while preserving shared support objects. Metrics and supports are
                # then prepared based on the experiment data.
                phase_runtime = prepare_phase_runtime(
                    phase,
                    experiment_data,
                )
                # Ensure that phase runtime has objective function and optimiser set
                assert phase_runtime.objective_function is not None
                assert phase_runtime.optimiser is not None

                # Initialise a PhaseResult to store the results of the current phase,
                # including solve results and refinement events.
                phase_result = PhaseResult(
                    phase_index=phase_index,
                    config=snapshot_phase_config(phase_index, phase),
                )

                # Resolve optional phase-start sensitivity weights before
                # prior-phase metrics are evaluated. This ensures baseline
                # and candidate objectives use the same frozen weights.
                phase_runtime.resolve_spatial_weighting(
                    identification_config.constitutive_law,
                    identification_config.parameters,
                    experiment_data,
                )

                # Evaluate current metrics on all previously identified phases.
                # phase results. These may not always be used, but can be used to
                # resolve baseline values to normalise metrics and / or for initialising DOFs.
                # Simpler to always compute for now.
                # The immediate predecessor is used only for EGI-informed seeding.
                previous_phases_metrics = (
                    phase_runtime.evaluate_previous_phases_metrics(
                        phase_index,
                        completed_phase_maps,
                        identification_config.constitutive_law,
                        experiment_data,
                        parameter_map_size,
                    )
                )

                # Resolve the baseline metric values for the current phase's objective function, if applicable.
                phase_runtime.resolve_objective_baseline(
                    previous_phases_metrics,
                )
                phase_runtime.resolve_refinement_baseline(
                    previous_phases_metrics,
                )

                # Initialise phase structure from parameter-map residuals.
                # This initialises the spatial parameterisations and their DOFs based on
                # the current parameter maps and any previous phase metrics.
                phase_runtime.initialise_parameterisation_structure(
                    identification_config.parameters,
                    parameter_map_size,
                    experiment_data,
                    previous_phases_metrics.get(phase_index - 1),
                )

                # Optionally screen the initial EGI-seeded basis before the
                # first joint phase solve. Accepted phase-start maps remain
                # frozen during screening; the winning basis is then restored
                # as active for the normal joint optimisation.
                if isinstance(
                    phase_runtime.refinement_policy,
                    EquilibriumGapBasisGrowthRefinement,
                ):
                    initial_screening_action = (
                        phase_runtime.refinement_policy.propose_initial_multistart(
                            phase_runtime,
                            phase_runtime.build_refinement_context(
                                identification_config.constitutive_law,
                                identification_config.parameters,
                                parameter_map_size,
                                experiment_data,
                                metrics=phase_runtime.metrics,
                                objective_function=phase_runtime.objective_function,
                            ),
                        )
                    )
                    if initial_screening_action is not None:
                        _apply_refinement_action(
                            phase_runtime,
                            phase_result,
                            phase_runtime.refinement_policy,
                            initial_screening_action,
                            phase_runtime.build_refinement_context(
                                identification_config.constitutive_law,
                                identification_config.parameters,
                                parameter_map_size,
                                experiment_data,
                                metrics=phase_runtime.metrics,
                                objective_function=phase_runtime.objective_function,
                            ),
                            experiment_data,
                        )

                # Prepare a phase-local constitutive law for optimisation,
                # which may include precomputed inputs for the radial return algorithm
                # to reduce repeated calculations (such as strain increments and
                # linear stress increments if applicable).
                # This is done once per phase, rather than once per solve iteration,
                # to avoid unnecessary recomputation.
                phase_constitutive_law = _prepare_phase_constitutive_law(
                    identification_config.constitutive_law,
                    phase_runtime,
                    experiment_data,
                    parameter_map_size,
                    phase.optimisation_newton_tolerance,
                    phase.cache_radial_return,
                )

                solve_iteration = 0
                while True:
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
                    optimisation_result = phase_runtime.optimiser.optimise(
                        phase_constitutive_law,
                        parameter_map_size,
                        phase_runtime.spatial_state.spatial_parameterisations,
                        phase_runtime.metrics,
                        phase_runtime.objective_function,
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
                        phase_runtime.optimiser,
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
                    solve_result.final_snapshot = snapshot_phase(
                        phase_runtime.spatial_parameterisations
                    )

                    # Store the resolved objective baseline in the solve result for logging
                    _record_objective_baseline(
                        solve_result,
                        phase_runtime.objective_function,
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
                        solve_result.accepted = True
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
                        metrics=phase_runtime.metrics,
                        objective_function=phase_runtime.objective_function,
                        objective_value=solve_result.final_objective.get("cost"),
                    )

                    # Check if the refinement policy proposes a refinement action based on the current phase runtime and context.
                    action = phase_runtime.refinement_policy.propose(
                        phase_runtime,
                        context,
                    )
                    if isinstance(
                        phase_runtime.refinement_policy,
                        EquilibriumGapBasisGrowthRefinement,
                    ):
                        solve_result.final_objective["combined_egi"] = (
                            phase_runtime.refinement_policy.last_combined_egi
                        )

                    # If no refinement action is proposed, break the loop and proceed to the next phase.
                    if action is None:
                        solve_result.accepted = True
                        _emit_solve_progress(
                            progress_callback,
                            kind="refinement_finished",
                            phase_index=phase_index,
                            phase_count=phase_count,
                            solve_iteration=solve_iteration,
                            message="no refinement proposed",
                        )
                        break

                    solve_result.accepted = action.accepts_current_solve

                    # Apply the proposed refinement action to the phase runtime and record it in the phase result
                    _apply_refinement_action(
                        phase_runtime,
                        phase_result,
                        phase_runtime.refinement_policy,
                        action,
                        context,
                        experiment_data,
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

                    # If the refinement action is terminal (e.g. RestoreBasisModelAction), update
                    # the parameter maps and break the loop to proceed to the next phase.
                    if action.terminal:
                        # Synchronise the phase runtime's parameter maps with the restored
                        # parameterisations after the terminal refinement action.
                        phase_runtime.update_constitutive_parameter_maps(
                            identification_config.parameters,
                            parameter_map_size,
                        )
                        break

                    # Reinitialise the phase runtime's DOFs from the updated parameter maps
                    phase_runtime.initialise_dofs(
                        identification_config.parameters,
                        parameter_map_size,
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
                completed_phase_maps[phase_index] = {
                    name: np.asarray(parameter.map, dtype=np.float64).copy()
                    for name, parameter in identification_config.parameters.items()
                }

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


def _prepare_phase_constitutive_law(
    constitutive_law: IConstitutiveLaw,
    phase_runtime: "PhaseRuntime",
    experiment_data: ExperimentData,
    parameter_map_size: np.ndarray,
    optimisation_newton_tolerance: float,
    cache_radial_return: bool,
) -> IConstitutiveLaw:
    """Build an optional phase-local fast constitutive evaluator.

    Constitutive laws without a preparation hook retain their existing
    behaviour. Known elastic maps are supplied only when both are represented
    by ``SpatialParameterisationKnown`` objects in this phase.
    """

    prepare = getattr(constitutive_law, "prepare_for_optimisation", None)
    if prepare is None:
        return constitutive_law

    # If both elastic modulus and Poisson's ratio are represented by known
    # spatial parameterisations, evaluate their maps and use them to compute
    # invariant quantities for the radial return algorithm.
    elastic_labels = (
        getattr(constitutive_law, "elastic_modulus_label", None),
        getattr(constitutive_law, "poissons_ratio_label", None),
    )
    fixed_elastic_parameter_maps = None
    if all(isinstance(label, str) for label in elastic_labels):
        labels = tuple(elastic_labels)
        if all(
            len(phase_runtime.spatial_parameterisations[label]) == 1
            and isinstance(
                phase_runtime.spatial_parameterisations[label][0],
                SpatialParameterisationKnown,
            )
            for label in labels
        ):
            all_maps = phase_runtime.spatial_state.evaluate_parameter_maps(
                parameter_map_size
            )
            fixed_elastic_parameter_maps = {
                label: all_maps[label]
                for label in labels
            }

    # Prepare a phase-local constitutive law with precomputed radial-return inputs.
    return prepare(
        experiment_data.strain,
        error_tolerance=optimisation_newton_tolerance,
        fixed_elastic_parameter_maps=fixed_elastic_parameter_maps,
        cache_radial_return=cache_radial_return,
    )


def _get_phase_reference_metrics(
    source_phase_index: int,
    phase_runtime: "PhaseRuntime",
    completed_phase_maps: dict[int, dict[str, np.ndarray]],
    reference_metric_results: dict[int, list[MetricResult]],
    constitutive_law: IConstitutiveLaw,
    experiment_data: ExperimentData,
    parameter_map_size: np.ndarray,
) -> list[MetricResult]:
    """Evaluate this phase's metrics on a completed phase, once."""

    if source_phase_index not in reference_metric_results:
        # Retrieve identified parameter maps for selected baseline phase
        source_maps = completed_phase_maps[source_phase_index]
        # Evaluate stress for selected baseline phase using its
        # identified parameter maps and the constitutive law.
        reference_stress = constitutive_law.calculate_stress(
            experiment_data.strain,
            source_maps,
        )
        # Global SBVFs do not provide spatial initialisation information and
        # cannot be constructed before this phase's parameterisations are
        # initialised. Preserve result alignment with an empty placeholder.
        reference_metric_results[source_phase_index] = []
        for metric in phase_runtime.metrics:
            if isinstance(
                metric,
                (SliceWiseForceReconstructionMetric, EquilibriumGapMetric),
            ):
                result = evaluate_metrics(
                    reference_stress,
                    constitutive_law,
                    parameter_map_size,
                    phase_runtime.spatial_parameterisations,
                    [metric],
                    experiment_data,
                    include_egi_diagnostics=True,
                )[0]
            else:
                result = MetricResult()
            reference_metric_results[source_phase_index].append(result)
    return reference_metric_results[source_phase_index]


def _map_rms(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(finite_values**2)))


def _seed_initial_basis_function(
    phase_runtime: "PhaseRuntime",
    basis: SpatialParameterisationBasisFunction,
    parameter: ConstitutiveParameter,
    experiment_data: ExperimentData,
    previous_phase_metric_results: list[MetricResult] | None,
    egi_smoothing_points: int,
) -> None:
    """Add one metric-informed or deterministic seed basis function."""

    centre = _previous_phase_egi_centre(
        phase_runtime,
        experiment_data,
        previous_phase_metric_results,
        egi_smoothing_points,
    )
    if centre is None:
        centre = (
            float(0.5 * (np.nanmin(basis.x) + np.nanmax(basis.x))),
            float(0.5 * (np.nanmin(basis.y) + np.nanmax(basis.y))),
        )

    kernel, height = _default_initial_basis(
        centre,
        basis,
        parameter,
    )

    assert basis.support.kernels is not None
    basis.support.kernels.append(kernel)
    for _, parameterisation in phase_runtime.get_parameterisations_using_support(
        basis.support,
    ):
        if not isinstance(
            parameterisation,
            SpatialParameterisationBasisFunction,
        ):
            continue
        parameterisation.heights.append(
            height if parameterisation is basis else None
        )


def _previous_phase_egi_centre(
    phase_runtime: "PhaseRuntime",
    experiment_data: ExperimentData,
    previous_phase_metric_results: list[MetricResult] | None,
    egi_smoothing_points: int = 3,
) -> tuple[float, float] | None:
    """Return the maximum smoothed previous-phase EGI location, if available."""

    if (
        previous_phase_metric_results is None
    ):
        return None

    egi_results = [
        result
        for metric, result in zip(
            phase_runtime.metrics,
            previous_phase_metric_results,
            strict=True,
        )
        if isinstance(metric, EquilibriumGapMetric)
    ]
    if not egi_results:
        return None

    policy = phase_runtime.refinement_policy
    if isinstance(policy, EquilibriumGapBasisGrowthRefinement):
        egi_map = policy.combine_egi_results(
            egi_results,
            phase_runtime.objective_function,
        ).combined_baseline_scaled_egi_map
    elif isinstance(
        phase_runtime.objective_function,
        CombinedForceAndEquilibriumGapObjective,
    ):
        objective = phase_runtime.objective_function
        egi_map = combine_equilibrium_gap_maps(
            egi_results,
            egi_baseline_values=objective.egi_baselines_for(len(egi_results)),
            window_weights=objective.egi_window_weights,
        )
    else:
        return None

    x = experiment_data.specimen_geometry.x
    y = experiment_data.specimen_geometry.y
    specimen_mask = (
        experiment_data.specimen_geometry.region_of_interest.sample_specimen_mask(
            x,
            y,
        )
    )
    smoothed_map = uniform_filter(
        np.where(np.isfinite(egi_map), egi_map, 0.0),
        size=egi_smoothing_points,
    )
    valid_support = uniform_filter(
        np.isfinite(egi_map).astype(float),
        size=egi_smoothing_points,
    )
    candidates = np.where(
        specimen_mask & (valid_support > 0.0),
        smoothed_map,
        np.nan,
    )
    if not np.any(np.isfinite(candidates)):
        return None

    row, column = np.unravel_index(
        np.nanargmax(candidates),
        candidates.shape,
    )
    return float(x[row, column]), float(y[row, column])


def _default_initial_basis(
    centre: tuple[float, float],
    basis: SpatialParameterisationBasisFunction,
    parameter: ConstitutiveParameter,
) -> tuple[BasisFunctionKernel, DegreeOfFreedom]:
    """Create one Gaussian with conventional initial DOFs."""

    x, y = basis.x, basis.y
    spacing = min(
        float(np.nanmedian(np.diff(x, axis=1))),
        float(np.nanmedian(np.diff(y, axis=0))),
    )
    diagonal = float(
        np.hypot(
            np.nanmax(x) - np.nanmin(x),
            np.nanmax(y) - np.nanmin(y),
        )
    )
    parameter_range = parameter.upper_bound - parameter.lower_bound
    minimum_variance = (3.0 * spacing) ** 2
    maximum_variance = max(
        diagonal**2,
        minimum_variance * (1.0 + 1.0e-6),
    )
    initial_variance = float(np.sqrt(minimum_variance * maximum_variance))
    min_x, max_x, min_y, max_y = basis.get_centre_bounds()

    return (
        basis.create_kernel(
            DegreeOfFreedom(centre[0], min_x, max_x),
            DegreeOfFreedom(centre[1], min_y, max_y),
            DegreeOfFreedom(
                initial_variance,
                minimum_variance,
                maximum_variance,
                scaling="log",
            ),
        ),
        DegreeOfFreedom(
            basis.initial_height_fraction * parameter_range,
            -parameter_range,
            parameter_range,
        ),
    )

def _record_objective_baseline(
    solve_result: SolveResult,
    objective_function: object,
) -> None:
    """Add resolved combined-objective baselines to durable solve diagnostics."""

    if isinstance(objective_function, CombinedForceAndEquilibriumGapObjective):
        solve_result.final_objective["baseline"] = (
            objective_function.baseline_diagnostics()
        )
        solve_result.final_objective["spatial_weighting"] = (
            objective_function.spatial_weighting_diagnostics()
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


def _apply_refinement_action(
    phase_runtime: "PhaseRuntime",
    phase_result: PhaseResult,
    refinement_policy: IRefinementPolicy,
    action,
    context: RefinementContext,
    experiment_data: ExperimentData,
) -> None:
    """Apply, prepare, and record one phase-local structural change."""

    target_before = _summarise_refinement_policy_target(
        phase_runtime,
        refinement_policy,
    )
    action.apply(phase_runtime, context)
    phase_runtime.prepare(experiment_data)
    target_after = _summarise_refinement_policy_target(
        phase_runtime,
        refinement_policy,
    )
    phase_result.refinement_events.append(
        RefinementEvent(
            event_index=len(phase_result.refinement_events),
            policy=snapshot_refinement_policy(refinement_policy),
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


@dataclass(slots=True)
class PhaseRuntime:
    """Prepared runtime state for one identification phase.

    The phase definition is a declarative configuration. This runtime object
    owns the working copies used during solving, including any prepared shared
    supports and prepared metric state.
    """

    spatial_parameterisations: dict[str, list[ISpatialParameterisation]]
    metrics: list[IMetric]
    objective_function: IObjectiveFunction | None = None
    optimiser: IOptimiser | None = None
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
        """Prepare shared supports and metrics for the current runtime state.

        The parameterisations may have changed, so this method ensures that the spatial state
        and metrics are consistent with the current parameterisations.

        parameterisations = the current unknown field definitions
        spatial state = the routing/indexing that knows which DOFs belong to which support or parameterisation
        metrics = the objective evaluation components
        prepare() = “make sure all three agree with the current experiment and current parameterisation layout”

        """

        # Update the spatial state (mapping of which DOFs belong to which support or parameterisation)
        # to reflect any changes to the spatial parameterisations
        self.rebuild_spatial_state()

        # Prepare any shared supports for the current runtime state so they are ready
        # for evaluation with the current experiment data
        self.spatial_state.prepare(experiment_data)

        # Prepare each metric for the current runtime state so they are ready for evaluation
        # with the current experiment data
        for metric in self.metrics:
            metric.initialise(experiment_data)

    def resolve_spatial_weighting(
        self,
        constitutive_law: IConstitutiveLaw,
        constitutive_parameters: dict[str, ConstitutiveParameter],
        experiment_data: ExperimentData,
    ) -> None:
        """Resolve optional objective weights from the accepted phase-start maps."""

        if not isinstance(
            self.objective_function,
            CombinedForceAndEquilibriumGapObjective,
        ):
            return
        if self.objective_function.spatial_weighting is None:
            return

        active_parameter_names = tuple(
            parameter_name
            for parameter_name, parameterisations in (
                self.spatial_parameterisations.items()
            )
            if any(
                not isinstance(parameterisation, SpatialParameterisationKnown)
                for parameterisation in parameterisations
            )
        )
        parameter_maps = {
            parameter_name: np.asarray(parameter.map, dtype=np.float64).copy()
            for parameter_name, parameter in constitutive_parameters.items()
        }
        self.objective_function.resolve_spatial_weights(
            constitutive_law=constitutive_law,
            parameter_maps=parameter_maps,
            active_parameter_names=active_parameter_names,
            metrics=self.metrics,
            experiment_data=experiment_data,
        )


    def initialise_parameterisation_structure(
        self,
        constitutive_parameters: dict[str, ConstitutiveParameter],
        size: np.ndarray,
        experiment_data: ExperimentData,
        previous_phase_metric_results: list[MetricResult] | None,
    ) -> None:
        """Initialise parameterisations sequentially from their residual maps."""

        parameter_map_relative_residual_tolerance = 0.01 # e.g. 0.01 is 1% of parameter range
        egi_smoothing_points = 3

        for parameter_name, parameterisations in self.spatial_parameterisations.items():
            parameter = constitutive_parameters[parameter_name]
            residual_map = np.asarray(parameter.map, dtype=np.float64).copy()
            parameter_range = parameter.upper_bound - parameter.lower_bound

            for parameterisation_index, parameterisation in enumerate(
                parameterisations,
            ):
                residual_parameter = ConstitutiveParameter(
                    residual_map,
                    (
                        parameter.lower_bound
                        if parameterisation_index == 0
                        else -parameter_range
                    ),
                    (
                        parameter.upper_bound
                        if parameterisation_index == 0
                        else parameter_range
                    ),
                )

                if (
                    isinstance(
                        parameterisation,
                        SpatialParameterisationBasisFunction,
                    )
                    and not parameterisation.kernels
                ):
                    residual_rms = _map_rms(residual_map)

                    if (
                        residual_rms
                        > parameter_map_relative_residual_tolerance
                        * parameter_range
                    ):
                        parameterisation.fit_to_map(
                            residual_map,
                            parameter_range=parameter_range,
                            max_basis_functions=parameterisation.initial_kernels_max,
                        )
                    else:
                        _seed_initial_basis_function(
                            self,
                            parameterisation,
                            parameter,
                            experiment_data,
                            previous_phase_metric_results,
                            egi_smoothing_points
                        )
                else:
                    parameterisation.initialise_from_constitutive_parameter(
                        residual_parameter,
                    )

                residual_map = residual_map - parameterisation.to_map(size)


    def initialise_dofs(
        self,
        constitutive_parameters: dict[str, ConstitutiveParameter],
        size: np.ndarray,
    ) -> None:
        """Initialise existing DOFs without changing parameterisation structure."""

        self.spatial_state.initialise_from_constitutive_parameters(
            constitutive_parameters,
            size,
        )

    def evaluate_previous_phases_metrics(
        self,
        phase_index: int,
        completed_phase_maps: dict[int, dict[str, np.ndarray]],
        constitutive_law: IConstitutiveLaw,
        experiment_data: ExperimentData,
        parameter_map_size: np.ndarray,
    ) -> dict[int, list[MetricResult]]:
        """Evaluate this phase's metrics on every completed earlier phase."""

        previous_phases_metrics: dict[int, list[MetricResult]] = {}
        for source_phase_index in range(phase_index):
            _get_phase_reference_metrics(
                source_phase_index,
                self,
                completed_phase_maps,
                previous_phases_metrics,
                constitutive_law,
                experiment_data,
                parameter_map_size,
            )
        return previous_phases_metrics

    def resolve_objective_baseline(
        self,
        previous_phases_metrics: dict[int, list[MetricResult]],
    ) -> None:
        """Resolve a prior-phase objective baseline, when configured."""

        if self.objective_function is None:
            raise RuntimeError("Phase runtime has no objective function.")

        # Metric baselines are currently only required for the combined force-and-equilibrium-gap objective function.
        if not isinstance(
            self.objective_function,
            CombinedForceAndEquilibriumGapObjective,
        ):
            return

        # If the objective function is not configured to use a prior-phase baseline, no action is needed.
        if (
            self.objective_function.baseline.mode
            is not CombinedObjectiveBaselineMode.PRIOR_PHASE
        ):
            return

        # Resolve prior phase index to be used for baselines
        source_phase_index = self.objective_function.baseline.phase_index
        if source_phase_index is None:
            raise RuntimeError("Validation did not provide a prior baseline phase.")

        # Resolve the metrics to be used for baselines
        try:
            metric_results = previous_phases_metrics[source_phase_index]
        except KeyError as error:
            raise RuntimeError(
                f"Prior baseline phase {source_phase_index} has not been evaluated."
            ) from error

        # Evaluate the baseline values from the defined phase metrics and store them
        # in the objective function for use during optimisation.
        self.objective_function.resolve_from_prior_phase(metric_results)

    def resolve_refinement_baseline(
        self,
        previous_phases_metrics: dict[int, list[MetricResult]],
    ) -> None:
        """Resolve prior-phase EGI scaling for non-combined objectives."""

        policy = self.refinement_policy
        if (
            not isinstance(policy, EquilibriumGapBasisGrowthRefinement)
            or isinstance(
                self.objective_function,
                CombinedForceAndEquilibriumGapObjective,
            )
        ):
            return
        source_phase_index = policy.baseline_phase_index
        if source_phase_index is None:
            raise RuntimeError("Validation did not provide an EGI baseline phase.")
        try:
            metric_results = previous_phases_metrics[source_phase_index]
        except KeyError as error:
            raise RuntimeError(
                f"Prior EGI baseline phase {source_phase_index} has not been evaluated."
            ) from error
        policy.resolve_from_prior_phase(self.metrics, metric_results)


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
            for parameter_name, old_parameterisations in (
                self.spatial_parameterisations.items()
            ):
                for old_parameterisation, new_parameterisation in zip(
                    old_parameterisations,
                    spatial_parameterisations[parameter_name],
                    strict=True,
                ):
                    if target is old_parameterisation:
                        self.refinement_policy.target = new_parameterisation
                        target = new_parameterisation
                        break
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
            parameter = constitutive_parameters[param_name]
            assembled_map = evaluate_parameterisations_to_map(
                parameterisation_list,
                parameter_map_size,
            )
            # Individually bounded additive parameterisations can still sum
            # outside the constitutive parameter's physical bounds. Keep the
            # accepted phase state feasible before it seeds a refinement or
            # subsequent phase.
            parameter.map = np.clip(
                assembled_map,
                parameter.lower_bound,
                parameter.upper_bound,
            )

    def build_refinement_context(
        self,
        constitutive_law: IConstitutiveLaw,
        constitutive_parameters: dict[str, ConstitutiveParameter],
        parameter_map_size: np.ndarray,
        experiment_data: ExperimentData,
        *,
        metrics: list[IMetric] | None = None,
        objective_function: object | None = None,
        objective_value: float | None = None,
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
            metrics=list(self.metrics if metrics is None else metrics),
            objective_function=objective_function,
            objective_value=objective_value,
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
    """Prepare phase runtime once experiment data are available.

    Validation has already checked that the phase configuration is legal.
    This step builds runtime working copies while preserving any support
    sharing declared by the caller, then prepares data-dependent support
    and metric state.
    """

    # Deep copy all mutable phase components together so configuration remains
    # declarative and shared runtime targets/supports retain their identity.
    (
        runtime_spatial_parameterisations,
        runtime_metrics,
        runtime_objective_function,
        runtime_optimiser,
        runtime_refinement_policy,
    ) = copy.deepcopy(
        (
            phase.spatial_parameterisations,
            phase.metrics,
            phase.objective_function,
            phase.optimiser,
            phase.refinement_policy,
        )
    )
    phase_runtime = PhaseRuntime(
        spatial_parameterisations=runtime_spatial_parameterisations,
        metrics=runtime_metrics,
        objective_function=runtime_objective_function,
        optimiser=runtime_optimiser,
        refinement_policy=runtime_refinement_policy,
    )

    # Prepare the phase runtime with experiment data, which may involve preparing shared supports and metrics.
    phase_runtime.prepare(experiment_data)
    return phase_runtime
