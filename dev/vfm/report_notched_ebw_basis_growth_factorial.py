"""Report the 2x2 Gaussian-kernel/basis-growth identification experiment."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from pyvale.vfm import (
    CombinedForceAndEquilibriumGapObjective,
    EquilibriumGapMetric,
    ExperimentData,
    SliceConfig,
    SliceWiseForceReconstructionMetric,
    load_identification_result,
)
from pyvale.vfm.postprocessing import (
    compute_plasticity_diagnostics,
    load_constitutive_law_from_result,
    load_known_parameter_maps,
)


DATASET = Path(
    "/home/robh/1_Projects/pyvale-vfm-test-data/notched-ebw/"
    "synthetic-fe/wdbn1-idealised-yield/pyvale-vfm"
)
RUN_ROOT = DATASET / "identification" / "prepared"
RUN_STAMP = "20260828_113526"
RUNS = {
    "A": RUN_ROOT / f"factorial_A_bivariate_egi_{RUN_STAMP}",
    "B": RUN_ROOT / f"factorial_B_spd_egi_{RUN_STAMP}",
    "C": RUN_ROOT / f"factorial_C_bivariate_sensitivity_{RUN_STAMP}",
    "D": RUN_ROOT / f"factorial_D_spd_sensitivity_{RUN_STAMP}",
}
LABELS = {
    "A": "A: conventional + EGI peak",
    "B": "B: SPD + EGI peak",
    "C": "C: conventional + sensitivity",
    "D": "D: SPD + sensitivity",
}
WINDOWS = (29, 57)
WINDOW_WEIGHTS = (29.0, 57.0)
FORCE_WEIGHT = 0.1


@dataclass(slots=True)
class RunSummary:
    case: str
    label: str
    kernel: str
    growth: str
    runtime_minutes: float
    evaluations: int
    solve_count: int
    accepted_solve_count: int
    basis_count: int
    active_dofs: int
    objective: float
    objective_gap_to_truth: float
    equilibrium_gap_cost: float
    force_cost: float
    egi_29: float
    egi_57: float
    force_rms: float
    yield_rmse_mpa: float
    yield_mae_mpa: float
    yielded_rmse_mpa: float
    unyielded_rmse_mpa: float
    yield_correlation: float
    hardening_mpa: float
    hardening_error_percent: float
    convergence: str
    yield_map: np.ndarray
    yield_error: np.ndarray
    solve_basis_counts: list[int]
    solve_costs: list[float]
    solve_accepted: list[bool]
    sensitivity_proposals: list[dict[str, object]]

    def serialisable(self) -> dict[str, object]:
        values = asdict(self)
        values.pop("yield_map")
        values.pop("yield_error")
        return values


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    experiment = ExperimentData.load_from_file(
        args.input / "experiment_data.yaml"
    )
    known = load_known_parameter_maps(args.input / "known_parameter_maps.npz")
    if known is None:
        raise RuntimeError("Known parameter maps are required.")
    mask = (
        experiment.specimen_geometry.region_of_interest.sample_specimen_mask(
            experiment.specimen_geometry.x,
            experiment.specimen_geometry.y,
        )
    )

    first_result = load_identification_result(args.runs["A"])
    law = load_constitutive_law_from_result(first_result)
    metrics = _create_metrics(experiment)
    baselines = _extract_baselines(first_result)
    truth_maps = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in known.items()
    }
    truth_stress = law.calculate_stress(experiment.strain, truth_maps)
    truth_objective, truth_components = _evaluate_objective(
        truth_stress,
        experiment,
        metrics,
        baselines,
    )
    plasticity = compute_plasticity_diagnostics(experiment, law, truth_maps)
    if plasticity is None:
        raise RuntimeError("Plasticity diagnostics are unavailable.")
    yielded = np.asarray(plasticity.yielded_datapoints, dtype=bool) & mask

    summaries = [
        _analyse_run(
            case,
            args.runs[case],
            experiment,
            known,
            mask,
            yielded,
            metrics,
            baselines,
            truth_objective,
        )
        for case in "ABCD"
    ]
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "truth_objective": truth_objective,
        "truth_components": truth_components,
        "truth_yielded_fraction": float(np.count_nonzero(yielded) / np.count_nonzero(mask)),
        "runs": [summary.serialisable() for summary in summaries],
        "factor_effects": _factor_effects(summaries),
    }
    (args.output / "factorial_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    _write_csv(args.output / "factorial_summary.csv", summaries)
    report = args.output / (
        "NOTCHED_EBW_BASIS_GROWTH_FACTORIAL_"
        f"{datetime.now().astimezone():%Y%m%d_%H%M_%Z}.pdf"
    )
    _write_report(
        report,
        summaries,
        experiment,
        known,
        mask,
        yielded,
        truth_objective,
        truth_components,
    )
    print(json.dumps({"report": str(report), **payload}, indent=2))


def _create_metrics(experiment: ExperimentData):
    egi = [
        EquilibriumGapMetric(window_size=(window, window))
        for window in WINDOWS
    ]
    force = SliceWiseForceReconstructionMetric(
        slice_config=SliceConfig(axis="x", num_slices=63)
    )
    for metric in [force, *egi]:
        metric.initialise(experiment)
    return force, egi


def _extract_baselines(result) -> tuple[np.ndarray, float]:
    accepted = [
        solve
        for solve in result.history.phases[-1].solve_results
        if solve.accepted
    ]
    components = accepted[-1].final_objective["components"]
    return (
        np.asarray(components["egi_baselines"], dtype=np.float64),
        float(components["force_baseline"]),
    )


def _evaluate_objective(stress, experiment, metrics, baselines):
    force, egi = metrics
    results = [
        force.evaluate_force_recon_error(stress, experiment).metric_result,
        *(metric.evaluate_equilibrium_gap(stress).metric_result for metric in egi),
    ]
    objective = CombinedForceAndEquilibriumGapObjective(
        force_weight=FORCE_WEIGHT,
        egi_window_weights=WINDOW_WEIGHTS,
        egi_baseline_values=baselines[0],
        force_baseline_value=baselines[1],
    )
    value = objective.evaluate(results)
    assert objective.last_result is not None
    components = {
        "equilibrium_gap_cost": objective.last_result.equilibrium_gap_cost,
        "force_cost": objective.last_result.force_cost,
        "egi_scalars": objective.last_result.egi_scalars.tolist(),
        "force_scalar": objective.last_result.force_scalar,
    }
    return value, components


def _analyse_run(
    case,
    path,
    experiment,
    known,
    mask,
    yielded,
    metrics,
    baselines,
    truth_objective,
) -> RunSummary:
    result = load_identification_result(path)
    phase = result.history.phases[-1]
    accepted = [solve for solve in phase.solve_results if solve.accepted]
    final_accepted = accepted[-1]
    stress = result.final_stress
    if stress is None:
        law = load_constitutive_law_from_result(result)
        stress = law.calculate_stress(experiment.strain, result.parameter_maps)
    objective, components = _evaluate_objective(
        stress,
        experiment,
        metrics,
        baselines,
    )
    saved_objective = float(final_accepted.final_objective["cost"])
    if not np.isclose(objective, saved_objective, rtol=2.0e-5, atol=1.0e-9):
        raise ValueError(
            f"Recomputed objective for {case} differs from saved value: "
            f"{objective} vs {saved_objective}."
        )

    snapshot = final_accepted.final_snapshot.spatial_parameterisations[
        "yield_strength"
    ][1]
    basis_count = int(snapshot.summary["num_kernels"])
    active_dofs = len(final_accepted.final_dofs)
    identified_yield = np.where(
        mask,
        result.parameter_maps["yield_strength"],
        np.nan,
    )
    true_yield = np.where(mask, known["yield_strength"], np.nan)
    error = identified_yield - true_yield
    unyielded = mask & ~yielded
    valid = np.isfinite(error)
    correlation = float(
        np.corrcoef(identified_yield[valid], true_yield[valid])[0, 1]
    )
    hardening = float(np.nanmean(result.parameter_maps["hardening_modulus"][mask]))
    true_hardening = float(np.nanmean(known["hardening_modulus"][mask]))

    solve_basis_counts = []
    solve_costs = []
    solve_accepted = []
    for solve in phase.solve_results:
        basis_snapshot = solve.final_snapshot.spatial_parameterisations[
            "yield_strength"
        ][1]
        solve_basis_counts.append(int(basis_snapshot.summary["num_kernels"]))
        solve_costs.append(float(solve.final_objective["cost"]))
        solve_accepted.append(bool(solve.accepted))
    proposals = []
    for event in phase.refinement_events:
        diagnostics = event.action.options.get("screening_diagnostics")
        if isinstance(diagnostics, dict) and (
            diagnostics.get("policy") == "sensitivity_correction"
        ):
            proposals.append(diagnostics)

    return RunSummary(
        case=case,
        label=LABELS[case],
        kernel="SPD" if case in "BD" else "conventional",
        growth="sensitivity" if case in "CD" else "EGI peak",
        runtime_minutes=float(result.metadata.run.runtime_seconds) / 60.0,
        evaluations=sum(
            int(solve.num_evaluations or 0)
            for history_phase in result.history.phases
            for solve in history_phase.solve_results
        ),
        solve_count=len(phase.solve_results),
        accepted_solve_count=len(accepted),
        basis_count=basis_count,
        active_dofs=active_dofs,
        objective=objective,
        objective_gap_to_truth=objective - truth_objective,
        equilibrium_gap_cost=float(components["equilibrium_gap_cost"]),
        force_cost=float(components["force_cost"]),
        egi_29=float(components["egi_scalars"][0]),
        egi_57=float(components["egi_scalars"][1]),
        force_rms=float(components["force_scalar"]),
        yield_rmse_mpa=float(np.sqrt(np.mean(error[valid] ** 2))),
        yield_mae_mpa=float(np.mean(np.abs(error[valid]))),
        yielded_rmse_mpa=float(np.sqrt(np.mean(error[yielded] ** 2))),
        unyielded_rmse_mpa=float(np.sqrt(np.mean(error[unyielded] ** 2))),
        yield_correlation=correlation,
        hardening_mpa=hardening,
        hardening_error_percent=100.0 * (hardening - true_hardening) / true_hardening,
        convergence=final_accepted.message,
        yield_map=identified_yield,
        yield_error=error,
        solve_basis_counts=solve_basis_counts,
        solve_costs=solve_costs,
        solve_accepted=solve_accepted,
        sensitivity_proposals=proposals,
    )


def _factor_effects(summaries):
    values = {summary.case: summary for summary in summaries}
    return {
        "spd_effect_with_egi": {
            "objective": values["B"].objective - values["A"].objective,
            "yield_rmse_mpa": values["B"].yield_rmse_mpa - values["A"].yield_rmse_mpa,
        },
        "spd_effect_with_sensitivity": {
            "objective": values["D"].objective - values["C"].objective,
            "yield_rmse_mpa": values["D"].yield_rmse_mpa - values["C"].yield_rmse_mpa,
        },
        "sensitivity_effect_conventional": {
            "objective": values["C"].objective - values["A"].objective,
            "yield_rmse_mpa": values["C"].yield_rmse_mpa - values["A"].yield_rmse_mpa,
        },
        "sensitivity_effect_spd": {
            "objective": values["D"].objective - values["B"].objective,
            "yield_rmse_mpa": values["D"].yield_rmse_mpa - values["B"].yield_rmse_mpa,
        },
    }


def _write_report(
    path,
    summaries,
    experiment,
    known,
    mask,
    yielded,
    truth_objective,
    truth_components,
):
    x = experiment.specimen_geometry.x
    y = experiment.specimen_geometry.y
    extent = [np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)]
    with PdfPages(
        path,
        metadata={
            "Title": "Notched EBW Gaussian basis-growth factorial",
            "Author": "PyVale diagnostic report",
            "CreationDate": datetime.now(),
        },
    ) as pdf:
        _title_page(pdf, summaries, truth_objective, truth_components, yielded, mask)
        _table_page(pdf, summaries, truth_objective)
        masked_truth = np.where(mask, known["yield_strength"], np.nan)
        _map_page(pdf, summaries, masked_truth, extent)
        _error_page(pdf, summaries, extent)
        _percentage_error_page(pdf, summaries, masked_truth, yielded, extent)
        _tradeoff_page(pdf, summaries, truth_objective)
        _history_page(pdf, summaries, truth_objective)
        _sensitivity_page(pdf, summaries, masked_truth, extent)
        _findings_page(pdf, summaries, truth_objective)


def _title_page(pdf, summaries, truth_objective, truth_components, yielded, mask):
    best_map = min(summaries, key=lambda item: item.yield_rmse_mpa)
    best_objective = min(summaries, key=lambda item: item.objective)
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.06, 0.89, "Notched EBW identification", fontsize=25, weight="bold")
    fig.text(0.06, 0.83, "Gaussian geometry × basis-growth factorial", fontsize=17)
    lines = [
        "Experiment",
        "• A/B isolate conventional versus SPD Gaussian coordinates under EGI-peak growth.",
        "• C/D isolate the same geometry choice under signed sensitivity-correction growth.",
        "• Scalar FRE+EGI objective and pattern-search optimiser are otherwise unchanged.",
        "• Results are recomputed from final saved stress/maps; no identification was rerun.",
        "",
        f"Truth objective: {truth_objective:.6f} "
        f"(EGI component {truth_components['equilibrium_gap_cost']:.4f}, "
        f"FRE component {truth_components['force_cost']:.4f}).",
        f"Truth-yielded specimen points: {100*np.count_nonzero(yielded)/np.count_nonzero(mask):.2f}%.",
        f"Lowest accepted objective: {best_objective.label} ({best_objective.objective:.6f}).",
        f"Lowest yield-map RMSE: {best_map.label} ({best_map.yield_rmse_mpa:.2f} MPa).",
    ]
    fig.text(0.08, 0.70, "\n".join(lines), fontsize=12, va="top", linespacing=1.5)
    fig.text(
        0.06,
        0.06,
        f"Generated {datetime.now().astimezone():%Y-%m-%d %H:%M %Z} from run stamp {RUN_STAMP}.",
        fontsize=9,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _table_page(pdf, summaries, truth_objective):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title("Final accepted results", fontsize=19, pad=18)
    headers = [
        "Case", "Kernel", "Growth", "Bases", "DOFs", "Time\n[min]", "Evals",
        "Objective", "Gap to\ntruth", "Yield RMSE\n[MPa]",
        "Yielded RMSE\n[MPa]", "Corr.", "H error\n[%]",
    ]
    rows = [
        [
            item.case, item.kernel, item.growth, item.basis_count, item.active_dofs,
            f"{item.runtime_minutes:.1f}", item.evaluations,
            f"{item.objective:.5f}", f"{item.objective_gap_to_truth:+.5f}",
            f"{item.yield_rmse_mpa:.2f}", f"{item.yielded_rmse_mpa:.2f}",
            f"{item.yield_correlation:.3f}", f"{item.hardening_error_percent:+.2f}",
        ]
        for item in summaries
    ]
    table = ax.table(cellText=rows, colLabels=headers, loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 2.15)
    for column in range(len(headers)):
        table[(0, column)].set_facecolor("#dbe9f4")
        table[(0, column)].set_text_props(weight="bold")
    ax.text(0.02, 0.48, f"Truth objective = {truth_objective:.6f}", fontsize=11, weight="bold")
    ax.text(
        0.02,
        0.43,
        "Objective proximity and material-map accuracy are reported separately: "
        "a lower scalar objective need not imply a better yield map.",
        fontsize=10,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _map_page(pdf, summaries, truth_yield, extent):
    maps = [("Truth", truth_yield), *[(item.case, item.yield_map) for item in summaries]]
    mask_values = np.concatenate([values[np.isfinite(values)] for _, values in maps])
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27), constrained_layout=True)
    for axis, (title, values) in zip(axes.flat, maps, strict=False):
        image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="viridis",
            vmin=float(np.min(mask_values)),
            vmax=float(np.max(mask_values)),
        )
        axis.set_title(title)
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
    axes.flat[-1].axis("off")
    fig.suptitle("Truth and final accepted yield-strength maps", fontsize=18)
    fig.colorbar(image, ax=axes, label="Yield strength [MPa]", shrink=0.8)
    pdf.savefig(fig)
    plt.close(fig)


def _error_page(pdf, summaries, extent):
    limit = max(np.nanpercentile(np.abs(item.yield_error), 99.5) for item in summaries)
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
    for axis, item in zip(axes.flat, summaries, strict=True):
        image = axis.imshow(
            item.yield_error,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_title(
            f"{item.case}: RMSE {item.yield_rmse_mpa:.1f} MPa; "
            f"yielded {item.yielded_rmse_mpa:.1f} MPa"
        )
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
    fig.suptitle("Final yield-strength error (identified − truth)", fontsize=18)
    fig.colorbar(image, ax=axes, label="Error [MPa]", shrink=0.82)
    pdf.savefig(fig)
    plt.close(fig)


def _percentage_error_page(pdf, summaries, truth_yield, yielded, extent):
    percentage_errors = [
        100.0 * item.yield_error / truth_yield
        for item in summaries
    ]
    limit = max(
        np.nanpercentile(np.abs(values), 99.5)
        for values in percentage_errors
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
    for axis, item, values in zip(
        axes.flat,
        summaries,
        percentage_errors,
        strict=True,
    ):
        valid = np.isfinite(values)
        global_rms = float(np.sqrt(np.mean(values[valid] ** 2)))
        yielded_rms = float(np.sqrt(np.mean(values[yielded] ** 2)))
        image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_title(
            f"{item.case}: RMS {global_rms:.1f}%; yielded {yielded_rms:.1f}%"
        )
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
    fig.suptitle(
        "Final yield-strength percentage error: "
        "100 × (identified − truth) / truth",
        fontsize=17,
    )
    fig.colorbar(image, ax=axes, label="Error [%]", shrink=0.82)
    pdf.savefig(fig)
    plt.close(fig)


def _tradeoff_page(pdf, summaries, truth_objective):
    labels = [item.case for item in summaries]
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
    plots = (
        ([item.objective for item in summaries], "Final scalar objective"),
        ([item.yield_rmse_mpa for item in summaries], "Yield-map RMSE [MPa]"),
        ([item.yielded_rmse_mpa for item in summaries], "Yielded-region RMSE [MPa]"),
        ([item.runtime_minutes for item in summaries], "Runtime [min]"),
    )
    colors = ("#5078a5", "#6c9f62", "#c57d56", "#8c6bb1")
    for axis, (values, title) in zip(axes.flat, plots, strict=True):
        bars = axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.bar_label(bars, fmt="%.3g", padding=3, fontsize=9)
    axes[0, 0].axhline(truth_objective, color="black", linestyle="--", label="truth")
    axes[0, 0].legend()
    fig.suptitle("Objective, physical accuracy, and cost", fontsize=18)
    pdf.savefig(fig)
    plt.close(fig)


def _history_page(pdf, summaries, truth_objective):
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
    for axis, item in zip(axes.flat, summaries, strict=True):
        accepted_x = [
            basis for basis, accepted in zip(item.solve_basis_counts, item.solve_accepted, strict=True)
            if accepted
        ]
        accepted_y = [
            cost for cost, accepted in zip(item.solve_costs, item.solve_accepted, strict=True)
            if accepted
        ]
        rejected_x = [
            basis for basis, accepted in zip(item.solve_basis_counts, item.solve_accepted, strict=True)
            if not accepted
        ]
        rejected_y = [
            cost for cost, accepted in zip(item.solve_costs, item.solve_accepted, strict=True)
            if not accepted
        ]
        axis.plot(accepted_x, accepted_y, "o-", label="accepted")
        axis.scatter(rejected_x, rejected_y, marker="x", s=70, color="tab:red", label="rejected trial")
        axis.axhline(truth_objective, color="black", linestyle="--", label="truth")
        axis.set_title(item.label)
        axis.set_xlabel("Basis count in solved trial")
        axis.set_ylabel("Objective")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Basis-growth history and acceptance decisions", fontsize=18)
    pdf.savefig(fig)
    plt.close(fig)


def _sensitivity_page(pdf, summaries, truth_yield, extent):
    sensitivity = [item for item in summaries if item.growth == "sensitivity"]
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27), constrained_layout=True)
    for axis, item in zip(axes, sensitivity, strict=True):
        axis.imshow(truth_yield, origin="lower", extent=extent, aspect="auto", cmap="Greys", alpha=0.7)
        for index, proposal in enumerate(item.sensitivity_proposals, start=2):
            centre = proposal["centre"]
            sign = float(proposal["proposed_sign"])
            axis.scatter(
                centre[0], centre[1], s=90, marker="v" if sign < 0 else "^",
                color="tab:blue" if sign < 0 else "tab:red", edgecolor="white",
            )
            axis.text(centre[0], centre[1], f" {index}", color="white", weight="bold")
        derivatives = [
            float(value["predicted_objective_directional_derivative"])
            for value in item.sensitivity_proposals
        ]
        axis.set_title(
            f"{item.case}: {len(item.sensitivity_proposals)} proposals; "
            f"predicted derivatives {', '.join(f'{value:.3g}' for value in derivatives)}"
        )
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
    fig.suptitle(
        "Sensitivity-correction proposal centres over the truth map\n"
        "down-triangle = reduce yield; up-triangle = increase yield",
        fontsize=16,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _findings_page(pdf, summaries, truth_objective):
    values = {item.case: item for item in summaries}
    objective_rank = sorted(summaries, key=lambda item: item.objective)
    map_rank = sorted(summaries, key=lambda item: item.yield_rmse_mpa)
    yielded_rank = sorted(summaries, key=lambda item: item.yielded_rmse_mpa)
    effects = _factor_effects(summaries)
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.06, 0.91, "Findings and implications", fontsize=22, weight="bold")
    lines = [
        f"• Objective ranking: {' < '.join(f'{item.case} ({item.objective:.5f})' for item in objective_rank)}; truth is {truth_objective:.5f}.",
        f"• Yield-map ranking: {' < '.join(f'{item.case} ({item.yield_rmse_mpa:.1f} MPa)' for item in map_rank)}.",
        f"• Yielded-region ranking: {' < '.join(f'{item.case} ({item.yielded_rmse_mpa:.1f} MPa)' for item in yielded_rank)}.",
        f"• SPD effect under EGI growth (B−A): objective {effects['spd_effect_with_egi']['objective']:+.5f}, RMSE {effects['spd_effect_with_egi']['yield_rmse_mpa']:+.1f} MPa.",
        f"• SPD effect under sensitivity growth (D−C): objective {effects['spd_effect_with_sensitivity']['objective']:+.5f}, RMSE {effects['spd_effect_with_sensitivity']['yield_rmse_mpa']:+.1f} MPa.",
        f"• Sensitivity effect with conventional geometry (C−A): objective {effects['sensitivity_effect_conventional']['objective']:+.5f}, RMSE {effects['sensitivity_effect_conventional']['yield_rmse_mpa']:+.1f} MPa.",
        f"• Sensitivity effect with SPD geometry (D−B): objective {effects['sensitivity_effect_spd']['objective']:+.5f}, RMSE {effects['sensitivity_effect_spd']['yield_rmse_mpa']:+.1f} MPa.",
        "",
        "Interpretation",
        "• A representation or growth rule is only a genuine improvement here if physical map error improves consistently, not merely the scalar objective.",
        "• Sensitivity proposals all passed the local descent-direction check; this validates the adjoint/sign plumbing, but does not guarantee that the fitted Gaussian is globally useful after re-optimisation.",
        "• Any disagreement between objective and RMSE rankings reinforces the earlier watering-down result: the current scalar remains an imperfect proxy for spatial yield-map fidelity.",
        "• Sensitivity growth improved conventional global RMSE, but worsened yielded-region RMSE in both geometry pairings; this is the more important caution for physical identification.",
        "• This is one deterministic synthetic case. Repeat the most promising pair across optimiser seeds before changing production defaults.",
        "",
        "Recommended next step",
        f"Treat {map_rank[0].case} as the global-RMSE lead, {yielded_rank[0].case} as the yielded-region lead, and {objective_rank[0].case} as the objective lead. Run seed replication before selecting a default.",
    ]
    fig.text(0.075, 0.84, "\n".join(lines), fontsize=10.8, va="top", linespacing=1.48)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path, summaries):
    rows = [summary.serialisable() for summary in summaries]
    fieldnames = [
        name for name, value in rows[0].items()
        if not isinstance(value, (list, dict))
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATASET / "prepared")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dev/vfm/output/notched_ebw_basis_growth_factorial_20260828"),
    )
    for case in "ABCD":
        parser.add_argument(
            f"--run-{case.lower()}",
            type=Path,
            default=RUNS[case],
        )
    args = parser.parse_args()
    args.runs = {case: getattr(args, f"run_{case.lower()}") for case in "ABCD"}
    return args


if __name__ == "__main__":
    main()
