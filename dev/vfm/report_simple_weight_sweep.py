"""Create a consolidated diagnostic PDF for the simple-objective weight sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from pyvale.vfm import ExperimentData, load_identification_result
from pyvale.vfm.postprocessing import load_known_parameter_maps
import report_notched_ebw_data_driven_identification as common


CASES = (
    ("01", "Baseline", "simple_baseline_7bf_workstation_20260830"),
    ("02", "Local EGI", "simple_weight_sweep_7bf_ws_remaining_20260830_02_local_egi"),
    ("03", "Balanced guards", "simple_weight_sweep_7bf_ws_remaining_20260830_03_balanced_guards"),
    ("04", "Half guards", "simple_weight_sweep_7bf_ws_remaining_20260830_04_half_guards"),
    ("05", "Near equal", "simple_weight_sweep_7bf_ws_remaining_20260830_05_near_equal"),
    ("06", "FRE emphasis", "simple_weight_sweep_7bf_ws_remaining_20260830_06_fre_emphasis"),
    ("07", "Broad EGI emphasis", "simple_weight_sweep_7bf_ws_remaining_20260830_07_broad_egi_emphasis"),
    ("08", "Strong FRE", "simple_weight_sweep_7bf_ws_remaining_20260830_08_strong_fre"),
    ("09", "Strong broad EGI", "simple_weight_sweep_7bf_ws_remaining_20260830_09_strong_broad_egi"),
    ("10", "Guards dominant", "simple_weight_sweep_7bf_ws_remaining_20260830_10_guards_dominant"),
)


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    experiment = ExperimentData.load_from_file(args.input / "experiment_data.yaml")
    known = load_known_parameter_maps(args.input / "known_parameter_maps.npz")
    truth = np.asarray(known["yield_strength"], dtype=float)
    mask = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(
        experiment.specimen_geometry.x, experiment.specimen_geometry.y
    )
    records = [
        _load_record(case_id, label, run_name, args, experiment, known, truth, mask)
        for case_id, label, run_name in CASES
    ]
    summary = _build_summary(records)

    with PdfPages(args.output, metadata={"Title": "Simple EGI objective weight sweep"}) as pdf:
        _overview_page(pdf, summary)
        _design_page(pdf, records)
        _recovery_page(pdf, records)
        _balance_page(pdf, records)
        _reference_page(pdf, records[0], truth, mask, experiment)
        _final_map_pages(pdf, records, truth, mask, experiment)
        _basis_page(pdf, records, mask, experiment)
        _recommendations_page(pdf, summary)

    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"pdf": str(args.output), "json": str(args.output.with_suffix('.json'))}, indent=2))


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--identification-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, default=Path("dev/vfm/output"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_record(case_id, label, run_name, args, experiment, known, truth, mask):
    run_dir = args.identification_root / run_name
    result_path = run_dir / "identification_result.yaml"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = load_identification_result(result_path)
    states = common._states(result, experiment, known)
    state_rows = []
    for state in states:
        values = np.asarray(state["maps"]["yield_strength"], dtype=float)
        state_rows.append({
            "state": state["label"],
            "rmse_mpa": float(np.sqrt(np.mean((values[mask] - truth[mask]) ** 2))),
            "minimum_mpa": float(np.min(values[mask])),
            "maximum_mpa": float(np.max(values[mask])),
        })
    solve = result.history.phases[1].solve_results[-1]
    final_objective = solve.final_objective or {}
    diagnostics = final_objective.get("objective_diagnostics", {})
    weights = diagnostics.get("objective_weights", {})
    components = final_objective.get("components", {})
    resolved_weights = {
        "informative_egi": float(weights["informative_egi"]),
        "fre_guard": float(weights["fre_guard"]),
        "broad_egi_guard": float(weights["broad_egi_guard"]),
    }
    raw = {
        "informative_egi": float(components["informative_egi_cost"]),
        "fre_guard": float(components["force_guard_cost"]),
        "broad_egi_guard": float(components["broad_guard_cost"]),
    }
    effective = {name: resolved_weights[name] * raw[name] for name in raw}
    snapshot = states[-1]["snapshot"]
    bf_extent = _basis_extent_summary(snapshot, common._extent(experiment))
    report_summary_path = args.report_root / run_name / "SIMPLE_SENSITIVITY_GATED_IDENTIFICATION.json"
    report_summary = json.loads(report_summary_path.read_text()) if report_summary_path.is_file() else {}
    return {
        "id": case_id,
        "label": label,
        "run_name": run_name,
        "states": state_rows,
        "final_map": np.asarray(states[-1]["maps"]["yield_strength"], dtype=float),
        "phase0_map": np.asarray(states[0]["maps"]["yield_strength"], dtype=float),
        "snapshot": snapshot,
        "weights": resolved_weights,
        "raw_components": raw,
        "effective_contributions": effective,
        "total_cost": float(components["total_cost"]),
        "supports": report_summary.get("selected_supports", {}),
        "gate": report_summary.get("gate", {}),
        "basis_extent": bf_extent,
    }


def _basis_extent_summary(snapshot, extent):
    xmin, xmax, ymin, ymax = extent
    widths = (xmax - xmin, ymax - ymin)
    maximum_ratios = [0.0, 0.0]
    outside = [False, False]
    count = 0
    for item in snapshot.spatial_parameterisations.get("yield_strength", []):
        if item.summary.get("kind") != "basis_functions":
            continue
        for kernel in item.summary.get("kernels", []):
            count += 1
            centre = np.asarray(kernel["centre"], dtype=float)
            radii = common._kernel_axis_radii(kernel)
            maximum_ratios[0] = max(maximum_ratios[0], 2.0 * radii[0] / widths[0])
            maximum_ratios[1] = max(maximum_ratios[1], 2.0 * radii[1] / widths[1])
            outside[0] |= centre[0] - radii[0] < xmin or centre[0] + radii[0] > xmax
            outside[1] |= centre[1] - radii[1] < ymin or centre[1] + radii[1] > ymax
    return {
        "count": count,
        "maximum_x_span_over_specimen": float(maximum_ratios[0]),
        "maximum_y_span_over_specimen": float(maximum_ratios[1]),
        "extends_beyond_x": bool(outside[0]),
        "extends_beyond_y": bool(outside[1]),
    }


def _build_summary(records):
    ranked = sorted(records, key=lambda item: item["states"][-1]["rmse_mpa"])
    baseline = records[0]
    baseline_components = baseline["raw_components"]
    rows = []
    for record in records:
        initial = record["states"][0]["rmse_mpa"]
        final = record["states"][-1]["rmse_mpa"]
        rows.append({
            "case": record["id"],
            "label": record["label"],
            "run_name": record["run_name"],
            "weights": record["weights"],
            "initial_rmse_mpa": initial,
            "final_rmse_mpa": final,
            "rmse_reduction_percent": 100.0 * (initial - final) / initial,
            "raw_components": record["raw_components"],
            "raw_component_ratios_to_baseline": {
                name: record["raw_components"][name] / baseline_components[name]
                for name in baseline_components
            },
            "effective_contributions": record["effective_contributions"],
            "total_cost": record["total_cost"],
            "basis_extent": record["basis_extent"],
        })
    return {
        "interpretation": "objective selection uses truth-free trade-offs; synthetic truth is held-out evaluation only",
        "runs_complete": len(records),
        "common_supports": baseline["supports"],
        "common_gate": baseline["gate"],
        "ranking_by_held_out_rmse": [
            {"case": item["id"], "label": item["label"], "final_rmse_mpa": item["states"][-1]["rmse_mpa"]}
            for item in ranked
        ],
        "recommended_next_default": {
            "case": baseline["id"],
            "label": baseline["label"],
            "weights": baseline["weights"],
            "reason": "best held-out recovery and a better guard compromise than the slightly more local-EGI-heavy case",
        },
        "next_steps": [
            "Retain 0.75 informative EGI / 0.15 FRE / 0.10 broad EGI as the provisional default.",
            "Replicate only the baseline, local-EGI and near-equal/guard-dominant brackets across noise seeds; do not repeat the full ten-case grid.",
            "Replace provisional scalar EGI noise scales with propagated support-specific noise before scientific claims.",
            "Investigate or constrain basis-function aspect ratios because final kernels extend substantially beyond the specimen in y.",
            "Use BF5 as an efficiency checkpoint and retain BF6-BF7 only when their incremental map/objective improvement is repeatable.",
        ],
        "cases": rows,
    }


def _overview_page(pdf, summary):
    ranking = summary["ranking_by_held_out_rmse"]
    best = ranking[0]
    second = ranking[1]
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(.055, .93, "Seven-BF objective-weight sweep: consolidated findings", fontsize=20, weight="bold")
    lines = [
        "Outcome: all ten configurations completed with identical EGI supports, gate, seed and solver budget; only objective weights changed.",
        "",
        f"Held-out synthetic evaluation: {best['label']} is best at {best['final_rmse_mpa']:.2f} MPa final yield-map RMSE; {second['label']} is second at {second['final_rmse_mpa']:.2f} MPa.",
        "The remaining guard-heavier configurations finish at 25.20–33.17 MPa. More FRE or broad-EGI weight does not improve recovery in this case.",
        "",
        "Truth-free interpretation",
        "  • Local-EGI emphasis obtains nearly the same informative residual but degrades both unmasked guards.",
        "  • The baseline retains the strongest recovery while avoiding that guard sacrifice.",
        "  • Guard-heavy cases marginally reduce some guard residuals, but at a much larger informative-EGI and held-out-map penalty.",
        "",
        "Decision",
        "  Retain 0.75 informative EGI / 0.15 FRE / 0.10 broad EGI as the provisional next default.",
        "  Do not treat this single synthetic specimen as final tuning evidence: replicate a reduced three-case bracket across noise realisations.",
        "",
        "Important limitation: the EGI noise scales remain provisional scalar values. These results compare algorithm behaviour, not final scientific uncertainty.",
    ]
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=112, subsequent_indent="    ") or [""])
    fig.text(.07, .85, "\n".join(wrapped), va="top", fontsize=11, linespacing=1.38)
    pdf.savefig(fig); plt.close(fig)


def _design_page(pdf, records):
    fig, axis = plt.subplots(figsize=(11.69, 8.27))
    axis.axis("off")
    columns = ["Case", "Purpose", "Informative EGI", "FRE", "Broad EGI", "Final RMSE [MPa]"]
    rows = [[
        record["id"], record["label"],
        f"{record['weights']['informative_egi']:.2f}",
        f"{record['weights']['fre_guard']:.2f}",
        f"{record['weights']['broad_egi_guard']:.2f}",
        f"{record['states'][-1]['rmse_mpa']:.2f}",
    ] for record in records]
    table = axis.table(cellText=rows, colLabels=columns, cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(9.5); table.scale(1.0, 1.6)
    for column in range(len(columns)):
        table[0, column].set_facecolor("#dce7ef"); table[0, column].set_text_props(weight="bold")
    axis.set_title(
        "Controlled sweep design\nCommon supports: fine 3×3, middle 9×9, broad 31×31; common frozen minimum→q90 sensitivity gate",
        fontsize=16, pad=20,
    )
    pdf.savefig(fig); plt.close(fig)


def _recovery_page(pdf, records):
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), constrained_layout=True)
    for record in records:
        values = [row["rmse_mpa"] for row in record["states"]]
        axes[0].plot(range(len(values)), values, marker="o", linewidth=1.2, markersize=3, label=f"{record['id']} {record['label']}")
    axes[0].set(title="Yield-map recovery through refinement", xlabel="Number of basis functions", ylabel="Held-out RMSE [MPa]")
    axes[0].grid(alpha=.25); axes[0].legend(fontsize=7.5, ncol=2)

    ranked = sorted(records, key=lambda item: item["states"][-1]["rmse_mpa"])
    labels = [item["id"] for item in ranked]
    values = [item["states"][-1]["rmse_mpa"] for item in ranked]
    colors = ["tab:green" if item["id"] == "01" else "tab:blue" for item in ranked]
    axes[1].barh(labels[::-1], values[::-1], color=colors[::-1])
    for y, value in enumerate(values[::-1]):
        axes[1].text(value + .35, y, f"{value:.2f}", va="center", fontsize=9)
    axes[1].set(title="Final BF7 ranking (truth used only here)", xlabel="Yield-map RMSE [MPa]", ylabel="Case")
    axes[1].grid(axis="x", alpha=.25)
    fig.suptitle("Recovery performance and diminishing returns", fontsize=17)
    pdf.savefig(fig); plt.close(fig)


def _balance_page(pdf, records):
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27), constrained_layout=True)
    labels = [record["id"] for record in records]
    x = np.arange(len(records))
    names = ("informative_egi", "fre_guard", "broad_egi_guard")
    display = ("Informative EGI", "FRE guard", "Broad-EGI guard")
    baseline = records[0]["raw_components"]
    width = .25
    for index, (name, title) in enumerate(zip(names, display, strict=True)):
        ratios = [record["raw_components"][name] / baseline[name] for record in records]
        axes[0].bar(x + (index - 1) * width, ratios, width, label=title)
    axes[0].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[0].set(title="Achieved raw component costs relative to baseline", ylabel="Ratio to case 01")
    axes[0].set_xticks(x, labels); axes[0].legend(ncol=3); axes[0].grid(axis="y", alpha=.25)

    bottom = np.zeros(len(records))
    for name, title in zip(names, display, strict=True):
        values = np.asarray([record["effective_contributions"][name] for record in records])
        axes[1].bar(x, values, bottom=bottom, label=title)
        bottom += values
    axes[1].set(title="Actual weighted contributions to terminal objective", xlabel="Case", ylabel="Contribution")
    axes[1].set_xticks(x, labels); axes[1].legend(ncol=3); axes[1].grid(axis="y", alpha=.25)
    fig.suptitle("Truth-free objective trade-offs", fontsize=17)
    pdf.savefig(fig); plt.close(fig)


def _reference_page(pdf, baseline, truth, mask, experiment):
    extent = common._extent(experiment)
    values = (truth, baseline["phase0_map"])
    labels = ("Known synthetic truth (evaluation only)", "Phase 0 homogeneous result")
    vmin = float(np.nanpercentile(np.concatenate([item[mask] for item in values]), 1))
    vmax = float(np.nanpercentile(np.concatenate([item[mask] for item in values]), 99))
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
    for column, (item, label) in enumerate(zip(values, labels, strict=True)):
        image = axes[0, column].imshow(np.where(mask, item, np.nan), origin="lower", extent=extent, aspect="equal", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, column].set_title(label)
    error = 100.0 * (baseline["phase0_map"] - truth) / truth
    limit = max(1.0, float(np.nanpercentile(np.abs(error[mask]), 99)))
    error_image = axes[1, 1].imshow(np.where(mask, error, np.nan), origin="lower", extent=extent, aspect="equal", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[1, 1].set_title("Phase 0 signed error [%]")
    axes[1, 0].axis("off")
    for axis in (axes[0, 0], axes[0, 1], axes[1, 1]):
        axis.set(xlabel="x [mm]", ylabel="y [mm]")
    fig.colorbar(image, ax=axes[0, :], label="Yield strength [MPa]", shrink=.8)
    fig.colorbar(error_image, ax=axes[1, 1], label="Error [%]", shrink=.8)
    fig.suptitle("Common starting point for all ten weight cases", fontsize=17)
    pdf.savefig(fig); plt.close(fig)


def _final_map_pages(pdf, records, truth, mask, experiment):
    extent = common._extent(experiment)
    all_maps = [record["final_map"] for record in records]
    vmin = float(np.nanpercentile(np.concatenate([item[mask] for item in [truth, *all_maps]]), 1))
    vmax = float(np.nanpercentile(np.concatenate([item[mask] for item in [truth, *all_maps]]), 99))
    errors = [100.0 * (item - truth) / truth for item in all_maps]
    error_limit = max(1.0, float(np.nanpercentile(np.abs(np.concatenate([item[mask] for item in errors])), 99)))
    for start in range(0, len(records), 2):
        page_records = records[start:start + 2]
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
        for column, record in enumerate(page_records):
            values = record["final_map"]
            image = axes[0, column].imshow(np.where(mask, values, np.nan), origin="lower", extent=extent, aspect="equal", cmap="viridis", vmin=vmin, vmax=vmax)
            axes[0, column].set_title(f"Case {record['id']}: {record['label']} — yield map")
            error = 100.0 * (values - truth) / truth
            error_image = axes[1, column].imshow(np.where(mask, error, np.nan), origin="lower", extent=extent, aspect="equal", cmap="RdBu_r", vmin=-error_limit, vmax=error_limit)
            axes[1, column].set_title(f"Signed error; RMSE {record['states'][-1]['rmse_mpa']:.2f} MPa")
            for axis in axes[:, column]:
                axis.set(xlabel="x [mm]", ylabel="y [mm]")
        fig.colorbar(image, ax=axes[0, :len(page_records)], label="Yield strength [MPa]", shrink=.8)
        fig.colorbar(error_image, ax=axes[1, :len(page_records)], label="Error [%]", shrink=.8)
        fig.suptitle("Final BF7 yield-strength recovery", fontsize=17)
        pdf.savefig(fig); plt.close(fig)


def _basis_page(pdf, records, mask, experiment):
    chosen_ids = {"01", "02", "06", "10"}
    selected = [record for record in records if record["id"] in chosen_ids]
    extent = common._extent(experiment)
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
    for axis, record in zip(axes.ravel(), selected, strict=True):
        image = axis.imshow(
            np.where(mask, record["final_map"], np.nan), origin="lower",
            extent=extent, aspect="equal", cmap="viridis",
        )
        common._draw_bases(axis, record["snapshot"])
        common._apply_display_limits(axis, common._basis_display_limits(record["snapshot"], extent))
        axis.set_aspect("equal", adjustable="box")
        axis.set(title=f"Case {record['id']}: {record['label']}", xlabel="x [mm]", ylabel="y [mm]")
        fig.colorbar(image, ax=axis, shrink=.72)
    fig.suptitle("Representative final basis geometry (view clipped to specimen or 120% extent)", fontsize=16)
    pdf.savefig(fig); plt.close(fig)


def _recommendations_page(pdf, summary):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(.055, .93, "Recommended next investigation", fontsize=20, weight="bold")
    lines = [
        "Provisional algorithm setting",
        "  Keep case 01: 0.75 informative EGI / 0.15 FRE / 0.10 broad EGI.",
        "  Treat the result as a robust working hypothesis, not a fitted universal constant.",
        "",
        "Next compute campaign — reduced and targeted",
        "  1. Run only cases 01, 02 and one guard-heavier bracket (05 or 10) across multiple artificial-noise seeds.",
        "  2. Compare median/spread of truth-free raw component costs, gate coverage and BF geometry; use synthetic RMSE only as held-out validation.",
        "  3. Stop expanding the weight grid unless the ranking changes across seeds.",
        "",
        "Blocking method work before scientific interpretation",
        "  4. Propagate DIC noise separately through each selected normalised-EGI support; replace the provisional scalar noise entries.",
        "  5. Quantify whether the 3×3 / 9×9 / 31×31 selection is stable under that propagated noise.",
        "  6. Address oversized y-direction basis functions with a specimen-scaled shape prior/bound or a demonstrably equivalent parameterisation.",
        "",
        "Efficiency decision",
        "  7. Use BF5 as a checkpoint. Continue to BF7 only if the latest correction produces a repeatable reduction in informative residual without degrading FRE/broad closure.",
        "",
        "This sequence returns focus to the two central questions: which three EGI scales carry resolved complementary information, and how should sensitivity-gated local information be balanced against global stress-scale/equilibrium guards?",
    ]
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=112, subsequent_indent="    ") or [""])
    fig.text(.07, .85, "\n".join(wrapped), va="top", fontsize=10.8, linespacing=1.35)
    pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
