"""Summarise final maps from the clean/noisy objective campaign."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pyvale.vfm import load_identification_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    manifest = json.loads((root / "campaign_manifest.json").read_text(encoding="utf-8"))
    with (root / "analysis/state_metrics.csv").open(newline="", encoding="utf-8") as stream:
        metrics = list(csv.DictReader(stream))
    final_by_case = {
        row["case_name"]: row
        for row in metrics
        if row["source"] == "campaign" and row["is_final_accepted"] == "True"
    }
    summary = []
    for case in manifest["cases"]:
        row = final_by_case.get(case["name"])
        result_path = root / case["name"] / "identification_result.yaml"
        if row is None or not result_path.is_file():
            continue
        result = load_identification_result(result_path)
        accepted = [item for item in result.history.phases[-1].solve_results if item.accepted]
        actual_cost = float(accepted[-1].final_objective["cost"])
        summary.append({
            "objective": case["objective"],
            "condition": case["condition"],
            "noise_scale": case["noise_scale"],
            "windows": case["windows"],
            "weights": case["weights"],
            "force_weight": case["force_weight"],
            "sensitivity_weighting": case["sensitivity_weighting"],
            "basis_count": row["basis_count"],
            "optimised_cost": actual_cost,
            "common_clean_objective": float(row["objective"]),
            "roi_rmse_mpa": float(row["roi_rmse_mpa"]),
            "yielded_rmse_mpa": float(row["yielded_rmse_mpa"]),
            "high_plastic_rmse_mpa": float(row["high_plastic_rmse_mpa"]),
            "yielded_mape_percent": float(row["yielded_mape_percent"]),
            "yielded_above_10pct": float(row["yielded_above_10pct"]),
            "hardening_error_percent": float(row["hardening_error_percent"]),
        })
    output = root / "analysis/objective_noise_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    report = root / "analysis/OBJECTIVE_NOISE_REPORT.md"
    clean = {row["objective"]: row for row in summary if row["condition"] == "clean"}
    noisy = {row["objective"]: row for row in summary if row["condition"] == "noise"}
    lines = [
        "# Notched-EBW clean/noisy objective campaign",
        "",
        "All variants used SPD sensitivity-correction growth, zero training-cost gate, one matched optimiser seed and a seven-BF cap. Alternative objective costs are not numerically comparable; map errors and the recomputed common clean objective are comparable.",
        "",
        "| Objective | Clean yielded | Noisy yielded | Δ noise | Clean high-plastic | Noisy high-plastic |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in clean.keys() & noisy.keys():
        left, right = clean[name], noisy[name]
        lines.append(
            f"| {name} | {left['yielded_rmse_mpa']:.2f} | {right['yielded_rmse_mpa']:.2f} | "
            f"{right['yielded_rmse_mpa']-left['yielded_rmse_mpa']:+.2f} | "
            f"{left['high_plastic_rmse_mpa']:.2f} | {right['high_plastic_rmse_mpa']:.2f} |"
        )
    if summary:
        best_clean = min((row for row in summary if row["condition"] == "clean"), key=lambda row: row["yielded_rmse_mpa"])
        best_noise = min((row for row in summary if row["condition"] == "noise"), key=lambda row: row["yielded_rmse_mpa"])
        lines.extend([
            "", "## Lowest yielded-region error",
            f"- Clean: `{best_clean['objective']}` at {best_clean['yielded_rmse_mpa']:.2f} MPa.",
            f"- Noisy: `{best_noise['objective']}` at {best_noise['yielded_rmse_mpa']:.2f} MPa.",
            "", "These are single-seed exploratory results. They may nominate, but cannot validate, a production objective.",
        ])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(summary), "csv": str(output), "report": str(report)}, indent=2))


if __name__ == "__main__":
    main()
