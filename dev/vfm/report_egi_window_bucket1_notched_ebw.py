"""Create the Bucket-1 EGI-window decision report from completed runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from pyvale.vfm import ExperimentData, load_identification_result
from pyvale.vfm.postprocessing import (
    compute_plasticity_diagnostics,
    load_constitutive_law_from_result,
    load_known_parameter_maps,
)


ROOT = Path(
    "/home/robh/1_Projects/pyvale-vfm-test-data/notched-ebw/"
    "synthetic-fe/wdbn1-idealised-yield/pyvale-vfm"
)
OUTPUT = Path("dev/vfm/output/egi_window_bucket_1_20260827")
RUNS = {
    "29/57, smooth 3\n(baseline)": "egi_window_baseline_15500_20260827",
    "17, smooth 1\n(selected diagnostic)": "egi_window_selected_17_smooth1_15500_20260827",
    "17/29/57, smooth 3\n(additive candidate)": "egi_windows_17_29_57_smooth3_15500_20260827",
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics = _load_metrics()
    _plot_comparison(metrics, OUTPUT / "comparison.png")
    _write_report(metrics, OUTPUT / "REPORT.md")
    _write_pdf(metrics, OUTPUT / "BUCKET_1_EGI_WINDOW_DECISION_20260827.pdf")
    print(f"Saved report to {OUTPUT}")


def _load_metrics() -> list[dict[str, object]]:
    experiment = ExperimentData.load_from_file(ROOT / "prepared/experiment_data.yaml")
    known = load_known_parameter_maps(ROOT / "prepared/known_parameter_maps.npz", ROOT / "prepared")
    assert known is not None
    records = []
    for label, name in RUNS.items():
        result = load_identification_result(ROOT / "identification/prepared" / name / "identification_result.yaml")
        law = load_constitutive_law_from_result(result)
        plasticity = compute_plasticity_diagnostics(experiment, law, known)
        assert plasticity is not None
        identified = result.parameter_maps["yield_strength"]
        valid = plasticity.yielded_datapoints & np.isfinite(known["yield_strength"]) & np.isfinite(identified)
        error = identified[valid] - known["yield_strength"][valid]
        equivalent_plastic_strain = np.nanmax(plasticity.equivalent_plastic_strain, axis=0)
        high = equivalent_plastic_strain >= np.nanpercentile(equivalent_plastic_strain[plasticity.yielded_datapoints], 75)
        high &= np.isfinite(known["yield_strength"]) & np.isfinite(identified)
        hardening = float(np.nanmean(result.parameter_maps["hardening_modulus"]))
        known_hardening = float(np.nanmean(known["hardening_modulus"]))
        solves = result.history.phases[1].solve_results
        accepted_solves = [solve for solve in solves if solve.accepted]
        if not accepted_solves:
            raise ValueError(f"No accepted phase-1 solve in {name}.")
        components = accepted_solves[-1].final_objective.get("components", {})
        egi_scalars = np.asarray(components.get("egi_scalars", ()), dtype=float)
        egi_baselines = np.asarray(components.get("egi_baselines", ()), dtype=float)
        egi_weights = np.asarray(components.get("egi_window_weights", ()), dtype=float)
        normalised_egi = egi_scalars / egi_baselines
        force_weight = float(components.get("force_weight", np.nan))
        force_cost = float(components.get("force_cost", np.nan))
        records.append({
            "label": label,
            "run": name,
            "yield_rmse_mpa": float(np.sqrt(np.mean(error**2))),
            "yield_mape_percent": float(np.mean(np.abs(error / known["yield_strength"][valid])) * 100.0),
            "high_plastic_rmse_mpa": float(np.sqrt(np.mean((identified[high] - known["yield_strength"][high]) ** 2))),
            "hardening_mpa": hardening,
            "hardening_error_mpa": abs(hardening - known_hardening),
            "hardening_at_lower_bound": bool(np.isclose(hardening, 500.0)),
            "runtime_minutes": float(sum(solve.runtime_seconds for solve in solves) / 60.0),
            "normalised_egi": normalised_egi.tolist(),
            "egi_objective_contributions": (
                (1.0 - force_weight) * egi_weights * normalised_egi
            ).tolist(),
            "force_cost": force_cost,
            "force_objective_contribution": force_weight * force_cost,
            "solves": [
                {"accepted": solve.accepted, "status": solve.status, "evaluations": solve.num_evaluations, "cost": solve.final_objective.get("cost")}
                for solve in solves
            ],
        })
    return records


def _plot_comparison(records: list[dict[str, object]], path: Path) -> None:
    labels = [str(record["label"]) for record in records]
    columns = (
        ("yield_rmse_mpa", "Yielded yield RMSE [MPa]"),
        ("yield_mape_percent", "Yielded yield MAPE [%]"),
        ("high_plastic_rmse_mpa", "High-plastic yield RMSE [MPa]"),
        ("hardening_error_mpa", "Hardening abs. error [MPa]"),
        ("runtime_minutes", "Phase-1 runtime [min]"),
    )
    fig, axes = plt.subplots(1, len(columns), figsize=(15.2, 3.8), layout="constrained")
    for axis, (key, title) in zip(axes, columns, strict=True):
        values = [float(record[key]) for record in records]
        axis.bar(labels, values, color=plt.cm.tab10(np.arange(len(records))))
        axis.set(title=title)
        axis.tick_params(axis="x", labelrotation=35, labelsize=7)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_report(records: list[dict[str, object]], path: Path) -> None:
    baseline, selected, additive = records
    lines = [
        "# Bucket 1 — EGI-window decision",
        "",
        "## Decision",
        "",
        "This report compares the retained two-scale baseline, the rejected single-scale diagnostic, and the additive local-scale candidate. Total objective costs are not compared across configurations because their metric sets differ.",
        "",
        "The increased evaluation budget resolved the original convergence concern for the retained configuration: its accepted solves ended at `minimum_mesh_size` or `max_iterations`, not `max_evaluations`.",
        "",
        "## Matched 15,500-evaluation comparison",
        "",
        "| Configuration | Yielded RMSE [MPa] | Yielded MAPE [%] | High-plastic RMSE [MPa] | Hardening error [MPa] | Runtime [min] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(f"| {record['label'].replace(chr(10), ' ')} | {record['yield_rmse_mpa']:.2f} | {record['yield_mape_percent']:.2f} | {record['high_plastic_rmse_mpa']:.2f} | {record['hardening_error_mpa']:.2f} | {record['runtime_minutes']:.2f} |")
    lines.extend([
        "",
        "## Per-term objective diagnostics from the final accepted solve",
        "",
        "| Configuration | Normalized EGI terms | EGI contributions | FRE cost | FRE contribution | Hardening lower bound |",
        "|---|---|---|---:|---:|---|",
    ])
    for record in records:
        normalised = ", ".join(f"{value:.3f}" for value in record["normalised_egi"])
        contributions = ", ".join(f"{value:.3f}" for value in record["egi_objective_contributions"])
        lines.append(f"| {record['label'].replace(chr(10), ' ')} | {normalised} | {contributions} | {record['force_cost']:.3f} | {record['force_objective_contribution']:.3f} | {record['hardening_at_lower_bound']} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"The single 17-point EGI window increased yielded-region RMSE from {baseline['yield_rmse_mpa']:.2f} to {selected['yield_rmse_mpa']:.2f} MPa and forced hardening to its lower bound (500 MPa). Its metric-only peak stability therefore did not predict identification quality.",
        "",
        "The additive candidate is the direct test of whether a small EGI scale helps property-gradient identification while the two existing scales and FRE retain global constraint.",
        "",
        "## Decision criteria for the additive candidate",
        "",
        "- Improve yielded or high-plastic yield RMSE by at least 5%.",
        "- Do not worsen either yield metric by more than 2%, hardening error by more than 10%, or runtime by more than 30%.",
        "- Do not hit an optimisation limit or a hardening bound.",
        "",
    ])
    path.write_text("\n".join(lines))


def _write_pdf(records: list[dict[str, object]], path: Path) -> None:
    with PdfPages(path) as pdf:
        figure = plt.figure(figsize=(11.7, 8.3), layout="constrained")
        axis = figure.add_subplot()
        axis.axis("off")
        table = [[
            str(record["label"]).replace("\n", " "),
            f"{record['yield_rmse_mpa']:.2f}",
            f"{record['yield_mape_percent']:.2f}",
            f"{record['high_plastic_rmse_mpa']:.2f}",
            f"{record['hardening_error_mpa']:.2f}",
            f"{record['runtime_minutes']:.2f}",
        ] for record in records]
        axis.table(cellText=table, colLabels=("Configuration", "Yield RMSE", "Yield MAPE", "High-plastic RMSE", "Hardening error", "Runtime [min]"), loc="center", cellLoc="center")
        axis.text(0.5, 0.86, "Bucket 1 — EGI-window decision", ha="center", fontsize=18, weight="bold")
        axis.text(0.5, 0.23, "The additive 17/29/57 candidate tests local sensitivity while retaining the existing global EGI and FRE constraints. Objective values are not directly comparable across metric sets.", ha="center", va="center", wrap=True, fontsize=12)
        axis.text(0.5, 0.13, "Adopt only if it materially improves yielded or high-plastic yield error without degrading hardening, force reconstruction, convergence, or runtime beyond the predefined limits.", ha="center", va="center", wrap=True, fontsize=11)
        pdf.savefig(figure)
        plt.close(figure)
        image = plt.imread(OUTPUT / "comparison.png")
        figure, axis = plt.subplots(figsize=(11.7, 5.0), layout="constrained")
        axis.imshow(image)
        axis.axis("off")
        pdf.savefig(figure)
        plt.close(figure)


if __name__ == "__main__":
    main()
