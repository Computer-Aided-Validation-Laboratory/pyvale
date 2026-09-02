"""Run the nominated five-phase fixed-geometry identification workflow.

Phase 2 is built by the existing guarded spatial-yield caller and is always
run as a complete BF1--BF7 trajectory.  Model order is selected afterward by
the existing measurement-noise replay rule; Phases 3--5 use production SBVF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyvale.vfm import (
    ConsoleProgressReporter,
    ConstitutiveParameter,
    DegreeOfFreedom,
    ExperimentData,
    IdentificationConfig,
    IdentificationPhase,
    MetricSBVF,
    OptimiserLeastSquares,
    SolveCheckpointWriter,
    VectorFirstResultPassthrough,
    VfmRegionOfInterest,
    load_identification_result,
    run_identification,
)
from pyvale.vfm.campaignprogress import atomic_write_json
from pyvale.vfm.fivephaseworkflow import (
    active_dof_summary,
    basis_amplitudes_from_snapshot,
    fixed_geometry_state_from_snapshot,
    make_phase_3_parameterisations,
    make_phase_4_parameterisations,
    make_phase_5_parameterisations,
    selected_phase_2_snapshot,
    snapshot_active_dof_summary,
)
from pyvale.vfm.modelorder import basis_count_from_stage
from pyvale.vfm.postprocessing import evaluate_snapshot_parameter_maps

import audit_wdbn1_guarded_x2 as guarded_audit
import call_notched_ebw_bivariate_identification as phase_2_runner


PHASE_12_DIRECTORY = "phase_1_2_yield_discovery"
PHASE_DIRECTORIES = {
    1: "phase_1_homogeneous_sbvf",
    3: "phase_3_homogeneous_hardening",
    4: "phase_4_spatial_hardening",
    5: "phase_5_joint_reconciliation",
}
WORKFLOW_STATE = "five_phase_workflow.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sbvf-max-evaluations",
        type=int,
        default=500,
        help="Maximum least-squares evaluations in each of Phases 3--5.",
    )
    parser.add_argument(
        "--phase-2-fix-hardening",
        action="store_true",
        help=(
            "Keep the Phase-1 homogeneous hardening value fixed during "
            "Phase 2. By default homogeneous hardening remains active."
        ),
    )
    args = phase_2_runner._parse_args(
        parser,
        default_max_basis_functions=7,
        default_fixed_basis_trajectory=True,
    )
    if args.sbvf_max_evaluations < 1:
        parser.error("--sbvf-max-evaluations must be positive.")
    _validate_five_phase_args(parser, args)

    output_dir = args.output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_file = (
        args.input / "experiment_data.yaml" if args.input.is_dir() else args.input
    )
    experiment = ExperimentData.load_from_file(experiment_file)
    if args.fre_region_of_interest is not None:
        experiment.specimen_geometry.force_reconstruction_region_of_interest = (
            VfmRegionOfInterest.from_yaml(
                args.fre_region_of_interest.expanduser().resolve()
            )
        )
    noise_diagnostics = phase_2_runner._apply_artificial_noise(experiment, args)
    law = phase_2_runner._create_constitutive_law(
        args.stress_backend,
        minimum_yield_strength=phase_2_runner.YIELD_BOUNDS_MPA[0],
    )
    shape = np.asarray(experiment.specimen_geometry.x.shape, dtype=np.uint32)
    reporter = ConsoleProgressReporter().report if args.show_progress else None

    manifest_path = output_dir / WORKFLOW_STATE
    manifest = _load_manifest(manifest_path, experiment_file, noise_diagnostics)

    phase_12_dir = output_dir / PHASE_12_DIRECTORY
    phase_1 = _homogeneous_phase(args)
    phase_2, _ = _create_phase_2(args, experiment, phase_12_dir)
    result_12 = _load_result_if_complete(phase_12_dir)
    if result_12 is None:
        _require_clean_incomplete_directory(phase_12_dir)
        parameters = _parameters(args, shape)
        result_12 = run_identification(
            experiment,
            IdentificationConfig(law, parameters, [phase_1, phase_2]),
            input_source=experiment_file,
            progress_callback=reporter,
            solve_checkpoint_callback=SolveCheckpointWriter(
                phase_12_dir / "solve_checkpoints"
            ),
        )
        result_12.save_to_yaml(phase_12_dir)
    result_1 = _persist_phase_1_result(
        result_12,
        experiment,
        law,
        phase_12_dir,
        output_dir / PHASE_DIRECTORIES[1],
    )
    manifest["phases"]["1"] = _phase_record(
        result_1,
        result_1.history.phases[0],
        output_dir / PHASE_DIRECTORIES[1],
        name="homogeneous_sbvf",
        configured_phase=phase_1,
    )
    manifest["phases"]["2"] = _phase_record(
        result_12,
        result_12.history.phases[1],
        phase_12_dir,
        name="spatial_yield_discovery",
        configured_phase=phase_2,
    )
    atomic_write_json(manifest_path, manifest)

    selection_path = output_dir / "phase_2_model_order_selection.json"
    if selection_path.is_file():
        selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    else:
        selection_payload = _run_model_order_selection(
            experiment,
            result_12,
            phase_12_dir,
        )
        atomic_write_json(selection_path, selection_payload)
    selected_count = basis_count_from_stage(
        selection_payload["selection"]["cumulative_selected"]
    )
    selected_snapshot = selected_phase_2_snapshot(
        result_12.history.phases[1], selected_count
    )
    geometry = fixed_geometry_state_from_snapshot(selected_snapshot)
    selected_maps = _complete_snapshot_maps(result_12, selected_snapshot, experiment)
    selected_maps_path = output_dir / "selected_phase_2_parameter_maps.npz"
    np.savez(selected_maps_path, **selected_maps)
    manifest["selected_phase_2"] = {
        **geometry.to_summary(),
        "selector": "cumulative_from_last_significant",
        "selection_file": str(selection_path),
        "parameter_maps": str(selected_maps_path),
    }
    atomic_write_json(manifest_path, manifest)

    phase_3 = _sbvf_phase(make_phase_3_parameterisations(
        float(np.nanmean(selected_maps["hardening_modulus"])),
        phase_2_runner.HARDENING_BOUNDS_MPA,
    ), args.sbvf_max_evaluations)
    result_3 = _run_single_phase(
        3,
        experiment,
        experiment_file,
        law,
        _parameters_from_maps(selected_maps),
        phase_3,
        output_dir,
        reporter,
    )
    homogeneous_h = float(np.nanmean(result_3.parameter_maps["hardening_modulus"]))
    manifest["phases"]["3"] = _phase_record(
        result_3, result_3.history.phases[0], output_dir / PHASE_DIRECTORIES[3],
        name="homogeneous_hardening_refit", configured_phase=phase_3,
    )
    atomic_write_json(manifest_path, manifest)

    phase_4 = _sbvf_phase(
        make_phase_4_parameterisations(
            geometry,
            experiment.specimen_geometry.x,
            experiment.specimen_geometry.y,
            hardening_amplitude_bound=(
                phase_2_runner.HARDENING_BOUNDS_MPA[1]
                - phase_2_runner.HARDENING_BOUNDS_MPA[0]
            ),
        ),
        args.sbvf_max_evaluations,
    )
    result_4 = _run_single_phase(
        4,
        experiment,
        experiment_file,
        law,
        _parameters_from_maps(result_3.parameter_maps),
        phase_4,
        output_dir,
        reporter,
    )
    hardening_amplitudes = basis_amplitudes_from_snapshot(
        result_4.history.phases[0].final_snapshot,
        "hardening_modulus",
    )
    manifest["phases"]["4"] = _phase_record(
        result_4, result_4.history.phases[0], output_dir / PHASE_DIRECTORIES[4],
        name="spatial_hardening_amplitudes", configured_phase=phase_4,
    )
    atomic_write_json(manifest_path, manifest)

    phase_5 = _sbvf_phase(
        make_phase_5_parameterisations(
            geometry,
            experiment.specimen_geometry.x,
            experiment.specimen_geometry.y,
            hardening_homogeneous=homogeneous_h,
            hardening_amplitudes=hardening_amplitudes,
            yield_bounds=phase_2_runner.YIELD_BOUNDS_MPA,
            hardening_bounds=phase_2_runner.HARDENING_BOUNDS_MPA,
        ),
        args.sbvf_max_evaluations,
    )
    result_5 = _run_single_phase(
        5,
        experiment,
        experiment_file,
        law,
        _parameters_from_maps(result_4.parameter_maps),
        phase_5,
        output_dir,
        reporter,
    )
    manifest["phases"]["5"] = _phase_record(
        result_5, result_5.history.phases[0], output_dir / PHASE_DIRECTORIES[5],
        name="fixed_geometry_joint_reconciliation", configured_phase=phase_5,
    )
    manifest["status"] = "complete"
    manifest["final_result"] = str(
        output_dir / PHASE_DIRECTORIES[5] / "identification_result.yaml"
    )
    atomic_write_json(manifest_path, manifest)
    print(json.dumps({
        "workflow": str(manifest_path),
        "selected_basis_count": selected_count,
        "final_result": manifest["final_result"],
    }, indent=2))


def _validate_five_phase_args(parser, args) -> None:
    if args.guarded_egi_objective_config is None:
        parser.error("Five-phase identification requires --guarded-egi-objective-config.")
    if args.fix_hardening:
        parser.error(
            "Use --phase-2-fix-hardening with the five-phase caller; "
            "Phase 1 must identify homogeneous H."
        )
    if args.egi_support_set != "fine-broad":
        parser.error("Five-phase Phase 2 requires --egi-support-set fine-broad.")
    if args.max_basis_functions != 7:
        parser.error("Five-phase Phase 2 requires --max-basis-functions 7.")
    if not args.fixed_basis_trajectory:
        parser.error("Five-phase Phase 2 requires --fixed-basis-trajectory.")


def _create_phase_2(args, experiment, output_dir):
    """Build production Phase 2 with configured homogeneous hardening status."""

    return phase_2_runner._create_spatial_yield_phase(
        args,
        experiment,
        output_dir,
        hardening_fixed=args.phase_2_fix_hardening,
    )


def _parameters(args, shape):
    return {
        "elastic_modulus": ConstitutiveParameter(
            args.elastic_modulus, 150_000.0, 250_000.0, shape
        ),
        "poissons_ratio": ConstitutiveParameter(
            args.poissons_ratio, 0.2, 0.4, shape
        ),
        "yield_strength": ConstitutiveParameter(
            args.initial_yield_strength,
            *phase_2_runner.YIELD_BOUNDS_MPA,
            shape,
        ),
        "hardening_modulus": ConstitutiveParameter(
            args.initial_hardening_modulus,
            *phase_2_runner.HARDENING_BOUNDS_MPA,
            shape,
        ),
    }


def _parameters_from_maps(maps):
    bounds = {
        "elastic_modulus": (150_000.0, 250_000.0),
        "poissons_ratio": (0.2, 0.4),
        "yield_strength": phase_2_runner.YIELD_BOUNDS_MPA,
        "hardening_modulus": phase_2_runner.HARDENING_BOUNDS_MPA,
    }
    return {
        name: ConstitutiveParameter(np.asarray(value).copy(), *bounds[name])
        for name, value in maps.items()
    }


def _homogeneous_phase(args):
    return IdentificationPhase(
        spatial_parameterisations={
            "elastic_modulus": [phase_2_runner.SpatialParameterisationKnown()],
            "poissons_ratio": [phase_2_runner.SpatialParameterisationKnown()],
            "yield_strength": [phase_2_runner.SpatialParameterisationHomogeneous(
                DegreeOfFreedom(
                    args.initial_yield_strength,
                    *phase_2_runner.YIELD_BOUNDS_MPA,
                )
            )],
            "hardening_modulus": [
                phase_2_runner.SpatialParameterisationHomogeneous(
                    DegreeOfFreedom(
                        args.initial_hardening_modulus,
                        *phase_2_runner.HARDENING_BOUNDS_MPA,
                    )
                )
            ],
        },
        metrics=[MetricSBVF(
            mesh_size=phase_2_runner.SBVF_MESH_SIZE,
            vf_scaling_fraction=phase_2_runner.SBVF_SCALING_FRACTION,
        )],
        objective_function=VectorFirstResultPassthrough(),
        optimiser=OptimiserLeastSquares(
            max_evaluations=args.phase_0_max_evaluations
        ),
    )


def _sbvf_phase(parameterisations, max_evaluations):
    return IdentificationPhase(
        spatial_parameterisations=parameterisations,
        metrics=[MetricSBVF(
            mesh_size=phase_2_runner.SBVF_MESH_SIZE,
            vf_scaling_fraction=phase_2_runner.SBVF_SCALING_FRACTION,
            perturbation_type="dof",
        )],
        objective_function=VectorFirstResultPassthrough(),
        optimiser=OptimiserLeastSquares(
            max_evaluations=max_evaluations,
            # Additive homogeneous+basis components are independently bounded
            # but can sum outside the physical map bounds during a trial.  The
            # accepted-state handoff already applies this projection; applying
            # the same projection to SBVF trial maps keeps the objective and
            # accepted state consistent.  Phase 2 uses its unchanged pattern
            # search path and is deliberately unaffected.
            parameter_map_bounds={
                "yield_strength": phase_2_runner.YIELD_BOUNDS_MPA,
                "hardening_modulus": phase_2_runner.HARDENING_BOUNDS_MPA,
            },
        ),
    )


def _run_single_phase(
    phase_number,
    experiment,
    experiment_file,
    law,
    parameters,
    phase,
    output_dir,
    reporter,
):
    phase_dir = output_dir / PHASE_DIRECTORIES[phase_number]
    existing = _load_result_if_complete(phase_dir)
    if existing is not None:
        return existing
    _require_clean_incomplete_directory(phase_dir)
    result = run_identification(
        experiment,
        IdentificationConfig(law, parameters, [phase]),
        input_source=experiment_file,
        progress_callback=reporter,
        solve_checkpoint_callback=SolveCheckpointWriter(
            phase_dir / "solve_checkpoints"
        ),
    )
    result.save_to_yaml(phase_dir)
    return result


def _persist_phase_1_result(result_12, experiment, law, checkpoint_root, target):
    existing = _load_result_if_complete(target)
    if existing is not None:
        return existing
    _require_clean_incomplete_directory(target)
    checkpoint = (
        checkpoint_root
        / "solve_checkpoints/phase_000_solve_000/identification_result.yaml"
    )
    if not checkpoint.is_file():
        raise RuntimeError(f"Completed Phase-1 checkpoint is missing: {checkpoint}")
    result = load_identification_result(checkpoint)
    result.final_stress = law.calculate_stress(
        experiment.strain, result.parameter_maps
    )
    result.save_to_yaml(target)
    return result


def _run_model_order_selection(experiment, result, run_dir):
    counts = [
        guarded_audit.common._basis_count(solve.final_snapshot)
        for solve in result.history.phases[1].solve_results
        if solve.final_snapshot is not None
    ]
    if counts != list(range(1, 8)):
        raise RuntimeError(
            "Phase 2 must persist exactly BF1 through BF7 before model-order "
            f"selection; found {counts}."
        )
    case = _selector_case(experiment, result, run_dir)
    _, transitions = guarded_audit._noise_replay(case)
    selection = guarded_audit._selection(transitions)
    return {
        "schema": "pyvale-five-phase-model-order-v1",
        "truth_used_for_selection": False,
        "complete_trajectory": [f"BF{value}" for value in range(1, 8)],
        "transitions": transitions,
        "selection": selection,
    }


def _selector_case(experiment, result, run_dir):
    law = guarded_audit.load_constitutive_law_from_result(result)
    mask = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(
        experiment.specimen_geometry.x,
        experiment.specimen_geometry.y,
    )
    snapshots = [("Phase0", result.history.phases[0].final_snapshot)]
    snapshots.extend(
        (f"BF{guarded_audit.common._basis_count(solve.final_snapshot)}", solve.final_snapshot)
        for solve in result.history.phases[1].solve_results
        if solve.final_snapshot is not None
    )
    states = []
    for label, snapshot in snapshots:
        maps = _complete_snapshot_maps(result, snapshot, experiment, mask=mask)
        states.append({"short": label, "stress_maps": maps, "maps": maps})
    metrics, supports, fre_spec = guarded_audit._metric_set(
        experiment, result, run_dir
    )
    gate, gate_diag = guarded_audit._frozen_gate(
        experiment, law, states[0]["stress_maps"]
    )
    preparations = guarded_audit._artifacts(run_dir, "guarded_egi_preparation")
    return {
        "label": "five-phase",
        "experiment": experiment,
        "result": result,
        "run": run_dir,
        "law": law,
        "states": states,
        "metrics": metrics,
        "supports": supports,
        "fre_spec": fre_spec,
        "gate": gate,
        "gate_diag": gate_diag,
        "prep": preparations,
    }


def _complete_snapshot_maps(result, snapshot, experiment, *, mask=None):
    if snapshot is None:
        raise ValueError("A completed phase snapshot is required.")
    if mask is None:
        mask = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(
            experiment.specimen_geometry.x,
            experiment.specimen_geometry.y,
        )
    maps = {
        name: np.asarray(value, dtype=np.float64).copy()
        for name, value in result.parameter_maps.items()
    }
    identified = evaluate_snapshot_parameter_maps(snapshot, experiment)
    for name, values in identified.items():
        values = np.asarray(values, dtype=np.float64)
        bounds = result.metadata.config.parameters[name]
        values = np.clip(values, bounds.lower_bound, bounds.upper_bound)
        reference = maps[name]
        maps[name] = np.where(mask, values, reference)
    return maps


def _phase_record(
    result,
    phase_result,
    directory,
    *,
    name,
    configured_phase=None,
):
    final_solve = phase_result.solve_results[-1]
    record = {
        "status": "complete",
        "name": name,
        "result": str(Path(directory) / "identification_result.yaml"),
        "parameter_maps": str(Path(directory) / "final_parameter_maps.npz"),
        "objective_diagnostics": final_solve.final_objective,
        "initial_dofs": final_solve.initial_dofs,
        "final_dofs": final_solve.final_dofs,
    }
    if configured_phase is not None:
        record["active_dofs"] = active_dof_summary(configured_phase)
    solve_states = []
    geometry_active = bool(
        configured_phase is not None
        and configured_phase.refinement_policy is not None
    )
    for solve in phase_result.solve_results:
        state = {
            "solve_iteration": solve.solve_iteration,
            "accepted": solve.accepted,
            "objective_diagnostics": solve.final_objective,
            "active_dof_count": len(solve.final_dofs),
            "active_dof_values": solve.final_dofs,
        }
        if solve.final_snapshot is not None:
            state["active_dofs"] = snapshot_active_dof_summary(
                solve.final_snapshot,
                include_geometry=geometry_active,
            )
            try:
                state["basis_count"] = fixed_geometry_state_from_snapshot(
                    solve.final_snapshot
                ).basis_count
            except ValueError:
                state["basis_count"] = 0
        solve_states.append(state)
    record["solve_states"] = solve_states
    snapshot = phase_result.final_snapshot
    if snapshot is not None:
        for parameter_name in ("yield_strength", "hardening_modulus"):
            try:
                record[f"{parameter_name}_amplitudes"] = list(
                    basis_amplitudes_from_snapshot(snapshot, parameter_name)
                )
            except ValueError:
                record[f"{parameter_name}_amplitudes"] = []
    return record


def _load_manifest(path, experiment_file, noise_diagnostics):
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("updated_at", None)
        if payload.get("input") != str(experiment_file):
            raise RuntimeError(
                "Existing five-phase workflow input does not match this run."
            )
        return payload
    return {
        "schema": "pyvale-five-phase-identification-v1",
        "status": "running",
        "input": str(experiment_file),
        "artificial_noise": noise_diagnostics,
        "phases": {},
    }


def _load_result_if_complete(directory):
    result_file = directory / "identification_result.yaml"
    return load_identification_result(result_file) if result_file.is_file() else None


def _require_clean_incomplete_directory(directory):
    if not directory.exists():
        return
    entries = list(directory.iterdir())
    if entries:
        checkpoints = directory / "solve_checkpoints/latest_accepted_checkpoint.json"
        detail = f" Latest accepted checkpoint: {checkpoints}." if checkpoints.exists() else ""
        raise RuntimeError(
            f"Incomplete phase output already exists at {directory}; it was preserved "
            f"and will not be overwritten.{detail}"
        )


if __name__ == "__main__":
    main()
