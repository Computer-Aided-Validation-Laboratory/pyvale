"""Create the Round-1 notched-EBW hybrid-objective decision report."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def main() -> None:
    args = _parse_args()
    root = args.round1.expanduser().resolve()
    screen = root / "screen"
    direct = root / "direct_fit"
    manifest = json.loads((screen / "screen_manifest.json").read_text())
    selections = json.loads((screen / "selected_objectives.json").read_text())
    windows = json.loads((screen / "selected_windows.json").read_text())
    scores = _read_csv(screen / "window_information_scores.csv")
    direct_rows = _read_csv(direct / "direct_fit_reference.csv") if (direct / "direct_fit_reference.csv").is_file() else []
    ranking = selections["ranking"]
    best = selections["raw"]
    lines = [
        "# Round 1 hybrid-objective decision",
        "",
        f"Generated from {manifest['states']} states and {manifest['noise_replicates']} WDBN1 noise realisations.",
        "",
        "## Frozen raw candidate",
        "",
        f"- Candidate: `{best['name']}`",
        f"- Alpha: {best['alpha']}",
        f"- EGI windows: {best['egi_windows']} pixels",
        f"- Physical supports: {windows['selected_labels']}",
        f"- Selection merit: {windows['selection_merit']:.3f}",
        "- Load frames are resolved from Phase-0 yielded fraction and stored in the objective JSON.",
        "- Feature floors are medians propagated from the configured WDBN1 noise realisations.",
        "",
        "## Projected candidate status",
        "",
        "The projected configuration is a gated candidate. Its semantic feature set is frozen, but it must not enter Round 2 until online native-DOF projection preparation and its BF-refresh integration test pass.",
        "",
        "## Candidate ranking",
        "",
        "| Rank | Candidate | Held-out merit |",
        "|---:|---|---:|",
    ]
    lines.extend(f"| {index} | `{row['name']}` | {row['merit']:.3f} |" for index, row in enumerate(ranking, 1))
    if direct_rows:
        lines.extend(["", "## Direct-fit representability", "", "| BF | ROI RMSE [MPa] | Yielded RMSE [MPa] | High-plastic RMSE [MPa] |", "|---:|---:|---:|---:|"])
        lines.extend(
            f"| {row['basis_count']} | {float(row['roi_rmse_mpa']):.3f} | {float(row['yielded_rmse_mpa']):.3f} | {float(row['high_plastic_rmse_mpa']):.3f} |"
            for row in direct_rows
        )
    lines.extend([
        "", "## Round-2 gate", "",
        "Proceed with the raw hybrid plus controls after reviewing this report. Include the projected hybrid only after its online preparation gate passes. Do not reinterpret the direct-fit curve as an attainable inverse result; it is the same-BF representability reference.",
        "", "Full evidence is in `screen/state_feature_rows.csv`, `screen/candidate_objective_scores.csv`, `screen/window_information_scores.csv`, and the two selected JSON files.",
    ])
    report = root / "ROUND1_DECISION.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _pdf(root / "ROUND1_DECISION.pdf", ranking, scores, direct_rows, lines)
    print(f"round1 report={report}", flush=True)


def _pdf(path, ranking, scores, direct_rows, lines):
    with PdfPages(path) as pdf:
        figure = plt.figure(figsize=(11.69, 8.27))
        figure.suptitle("Notched-EBW Round 1 objective decision", fontsize=18, y=0.95)
        summary = "\n".join(line for line in lines[2:18] if not line.startswith("|") and not line.startswith("##"))
        figure.text(0.07, 0.86, summary, va="top", fontsize=11, linespacing=1.5)
        pdf.savefig(figure); plt.close(figure)
        figure, axis = plt.subplots(figsize=(11.69, 8.27), constrained_layout=True)
        top = ranking[:9]
        axis.barh(np.arange(len(top)), [row["merit"] for row in top])
        axis.set_yticks(np.arange(len(top)), [row["name"] for row in top])
        axis.invert_yaxis(); axis.set(xlabel="Development/validation mean Spearman merit", title="Candidate ranking")
        axis.grid(axis="x", alpha=0.25); pdf.savefig(figure); plt.close(figure)
        if direct_rows:
            figure, axis = plt.subplots(figsize=(11.69, 8.27), constrained_layout=True)
            bf = [int(row["basis_count"]) for row in direct_rows]
            for field, label in (("roi_rmse_mpa", "ROI"), ("yielded_rmse_mpa", "Yielded"), ("high_plastic_rmse_mpa", "High plastic")):
                axis.plot(bf, [float(row[field]) for row in direct_rows], marker="o", label=label)
            axis.set(xlabel="SPD Gaussian BF count", ylabel="RMSE [MPa]", title="Direct-fit representability reference")
            axis.grid(alpha=0.25); axis.legend(); pdf.savefig(figure); plt.close(figure)


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round1", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
