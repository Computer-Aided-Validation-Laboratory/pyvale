"""Summarise the 2026-08-27 notched-EBW overnight geometry sweep."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyvale.vfm import ExperimentData, load_identification_result
from pyvale.vfm.postprocessing import (
    compute_plasticity_diagnostics,
    load_constitutive_law_from_result,
    load_known_parameter_maps,
)


DATASET = Path(
    "/home/robh/1_Projects/pyvale-vfm-test-data/notched-ebw/"
    "synthetic-fe/wdbn1-idealised-yield/pyvale-vfm"
)
PREFIX = "overnight_geometry_sweep_20260827_"
OUTPUT = Path("dev/vfm/output/overnight_geometry_sweep_20260827")
CONTROL = (
    DATASET
    / "identification/prepared/egi_window_baseline_15500_20260827/identification_result.yaml"
)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    experiment = ExperimentData.load_from_file(DATASET / "prepared/experiment_data.yaml")
    known = load_known_parameter_maps(DATASET / "prepared/known_parameter_maps.npz")
    if known is None:
        raise RuntimeError("Known parameter maps are required.")

    control = load_identification_result(CONTROL)
    law = load_constitutive_law_from_result(control)
    plasticity = compute_plasticity_diagnostics(experiment, law, known)
    if plasticity is None:
        raise RuntimeError("Known-map plasticity diagnostics are required.")
    yielded = np.asarray(plasticity.yielded_datapoints, dtype=bool)
    peak_plastic = np.nanmax(plasticity.equivalent_plastic_strain, axis=0)
    threshold = float(np.nanpercentile(peak_plastic[yielded], 75.0))
    high_plastic = yielded & (peak_plastic >= threshold)
    roi = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(
        experiment.specimen_geometry.x,
        experiment.specimen_geometry.y,
    )

    rows = [_result_row("retained_control", control, known, roi, yielded, high_plastic)]
    result_root = DATASET / "identification/prepared"
    for result_file in sorted(result_root.glob(f"{PREFIX}*/identification_result.yaml")):
        label = result_file.parent.name.removeprefix(PREFIX)
        rows.append(
            _result_row(
                label,
                load_identification_result(result_file),
                known,
                roi,
                yielded,
                high_plastic,
            )
        )
    rows.extend(_fixed_rows(known, roi, yielded, high_plastic))

    _write_csv(OUTPUT / "sweep_summary.csv", rows)
    _plot(rows, OUTPUT / "sweep_summary.png")
    (OUTPUT / "REPORT.md").write_text(_report(rows), encoding="utf-8")
    print(f"Saved report inputs to {OUTPUT}")


def _result_row(label, result, known, roi, yielded, high_plastic):
    phase = result.history.phases[1]
    accepted = [solve for solve in phase.solve_results if solve.accepted]
    final = accepted[-1]
    maps = result.parameter_maps
    basis = next(
        item
        for item in phase.final_snapshot.spatial_parameterisations["yield_strength"]
        if item.summary.get("kind") == "basis_functions"
    )
    error = np.asarray(maps["yield_strength"]) - known["yield_strength"]
    hardening = float(np.nanmean(maps["hardening_modulus"]))
    return {
        "case": label,
        "kind": "control" if label == "retained_control" else "free_geometry",
        "bases": int(basis.summary.get("num_kernels", 0)),
        "objective": float(final.final_objective["cost"]),
        "yielded_rmse_mpa": _rmse(error, yielded),
        "high_plastic_rmse_mpa": _rmse(error, high_plastic),
        "roi_rmse_mpa": _rmse(error, roi),
        "hardening_mpa": hardening,
        "hardening_abs_error_mpa": abs(hardening - 4000.0),
        "evaluations": sum(int(solve.num_evaluations or 0) for solve in phase.solve_results),
        "runtime_minutes": sum(float(solve.runtime_seconds or 0.0) for solve in phase.solve_results) / 60.0,
        "final_status": str(final.status),
    }


def _fixed_rows(known, roi, yielded, high_plastic):
    root = OUTPUT / "fixed_geometry"
    rows = []
    for case in ("A_oracle_geometry_hardening_fixed", "B_oracle_geometry_hardening_free"):
        with (root / case / "summary.json").open() as stream:
            summary = json.load(stream)
        with np.load(root / case / "result.npz") as bundle:
            error = bundle["yield_strength"] - known["yield_strength"]
        hardening = float(summary["hardening_mpa"])
        rows.append({
            "case": case,
            "kind": "oracle_fixed_geometry",
            "bases": 5,
            "objective": float(summary["objective"]),
            "yielded_rmse_mpa": _rmse(error, yielded),
            "high_plastic_rmse_mpa": _rmse(error, high_plastic),
            "roi_rmse_mpa": _rmse(error, roi),
            "hardening_mpa": hardening,
            "hardening_abs_error_mpa": abs(hardening - 4000.0),
            "evaluations": int(summary["evaluations"]),
            "runtime_minutes": np.nan,
            "final_status": str(summary["status"]),
        })
    return rows


def _rmse(error, mask):
    return float(np.sqrt(np.nanmean(np.asarray(error)[mask] ** 2)))


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _short(label):
    replacements = {
        "retained_control": "control",
        "bases4_seed0_smooth3_mesh010": "4B s0 sm3 m.10",
        "bases5_seed0_smooth3_mesh010": "5B s0 sm3 m.10",
        "bases5_seed1_smooth3_mesh010": "5B s1 sm3 m.10",
        "bases5_seed2_smooth3_mesh010": "5B s2 sm3 m.10",
        "bases5_seed3_smooth3_mesh010": "5B s3 sm3 m.10",
        "bases5_seed0_smooth1_mesh010": "5B s0 sm1 m.10",
        "bases5_seed0_smooth5_mesh010": "5B s0 sm5 m.10",
        "bases5_seed0_smooth3_mesh005": "5B s0 sm3 m.05",
        "bases5_seed0_smooth3_mesh020": "5B s0 sm3 m.20",
        "A_oracle_geometry_hardening_fixed": "oracle H fixed",
        "B_oracle_geometry_hardening_free": "oracle H free",
    }
    return replacements.get(label, label)


def _plot(rows, path):
    labels = [_short(row["case"]) for row in rows]
    colours = [
        "tab:green" if row["kind"] == "oracle_fixed_geometry"
        else "tab:gray" if row["kind"] == "control"
        else "tab:blue"
        for row in rows
    ]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    axes[0].bar(x, [row["objective"] for row in rows], color=colours)
    axes[0].set_ylabel("Mechanical objective")
    axes[1].bar(x, [row["yielded_rmse_mpa"] for row in rows], color=colours)
    axes[1].set_ylabel("Yielded-region RMSE [MPa]")
    axes[2].bar(x, [row["hardening_abs_error_mpa"] for row in rows], color=colours)
    axes[2].set_ylabel("|Hardening error| [MPa]")
    axes[2].set_xticks(x, labels, rotation=45, ha="right")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _report(rows):
    free = [row for row in rows if row["kind"] == "free_geometry"]
    control = next(row for row in rows if row["kind"] == "control")
    oracle = [row for row in rows if row["kind"] == "oracle_fixed_geometry"]
    best_truth = min(free, key=lambda row: row["yielded_rmse_mpa"])
    best_objective = min(free, key=lambda row: row["objective"])
    standard_seeds = [
        row for row in free
        if row["case"].startswith("bases5_seed")
        and row["case"].endswith("smooth3_mesh010")
    ]
    table = [
        "| Case | Bases | Objective | Yielded RMSE [MPa] | ROI RMSE [MPa] | |H error| [MPa] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {_short(row['case'])} | {row['bases']} | {row['objective']:.5f} | "
            f"{row['yielded_rmse_mpa']:.2f} | {row['roi_rmse_mpa']:.2f} | "
            f"{row['hardening_abs_error_mpa']:.1f} |"
        )
    seed_rmse = [row["yielded_rmse_mpa"] for row in standard_seeds]
    seed_objectives = [row["objective"] for row in standard_seeds]
    return "\n".join([
        "---",
        "title: \"Notched-EBW overnight geometry sweep\"",
        "date: \"28 August 2026\"",
        "geometry: margin=18mm",
        "---",
        "",
        "# Executive summary",
        "",
        (
            f"The sweep confirms that adding more freely optimized Gaussian bases does not close the identification gap. "
            f"The best free-geometry result by known yielded-region accuracy was **{_short(best_truth['case'])}**, "
            f"with {best_truth['yielded_rmse_mpa']:.2f} MPa RMSE and objective {best_truth['objective']:.5f}. "
            f"The best mechanical objective was **{_short(best_objective['case'])}** at {best_objective['objective']:.5f}, "
            f"but its yielded-region RMSE was {best_objective['yielded_rmse_mpa']:.2f} MPa."
        ),
        "",
        (
            f"Across the four nominally identical five-basis runs, changing only optimizer seed produced objectives from "
            f"{min(seed_objectives):.5f} to {max(seed_objectives):.5f} and yielded-region RMSE from "
            f"{min(seed_rmse):.2f} to {max(seed_rmse):.2f} MPa. This is substantial path dependence."
        ),
        "",
        (
            f"In contrast, the fixed oracle geometry converged to objectives of {min(row['objective'] for row in oracle):.5f}–"
            f"{max(row['objective'] for row in oracle):.5f} and yielded-region RMSE of "
            f"{min(row['yielded_rmse_mpa'] for row in oracle):.2f}–{max(row['yielded_rmse_mpa'] for row in oracle):.2f} MPa. "
            "Free hardening remained close to 4000 MPa. The basis representation and mechanical objective can therefore support "
            "a good solution when geometry is suitable; free geometry selection remains the limiting step."
        ),
        "",
        "# Results",
        "",
        *table,
        "",
        "![Sweep comparison.](sweep_summary.png){width=100%}",
        "",
        "# Interpretation",
        "",
        f"The retained three-basis control has objective {control['objective']:.5f} and yielded-region RMSE {control['yielded_rmse_mpa']:.2f} MPa. "
        "Moving to four or five bases can reduce the mechanical objective, but the truth error does not improve consistently. "
        "Some additional bases improve equilibrium while constructing compensating, broad or low-information features rather than recovering the yield-strength field.",
        "",
        "Changing EGI smoothing or the initial pattern-search mesh alters the final answer, but none approaches the oracle closure result. "
        "These controls affect the optimization path rather than correcting the underlying free-geometry ill-conditioning.",
        "",
        "The lower objective-improvement gate used here was diagnostic. It successfully forced controlled model growth, but the extra bases did not reliably improve the known synthetic map. "
        "This supports retaining the existing acceptance gate until geometry proposal and staged optimization are improved.",
        "",
        "# Recommended next step",
        "",
        "Implement an EGI-derived fixed-geometry growth experiment: propose centres independently from the 29- and 57-point EGI maps, derive initial widths and orientation from local EGI shape, and optimize only homogeneous yield, basis heights and hardening. "
        "Then unlock centres, widths and angles in separate blocks, accepting a block only when its objective improvement is repeatable and does not degrade validation. "
        "This directly targets the demonstrated failure mode while preserving the objective, EGI windows and parameter representation that have already passed closure tests.",
        "",
        "# Scope and definitions",
        "",
        "All free-geometry runs use EGI windows 29 and 57, FRE weight 0.1, five-point maximum model order unless stated otherwise, and a zero relative-improvement gate for controlled growth. "
        "Yielded and high-plastic masks are calculated once from the known synthetic parameter maps. The high-plastic mask is the upper quartile of peak equivalent plastic strain within known yielded points. Oracle geometry uses truth-derived fitted basis geometry and is diagnostic only.",
        "",
    ])


if __name__ == "__main__":
    main()
