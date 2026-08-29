"""Create a concise independent audit of the notched-EBW Round-1 bundle."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import textwrap

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def main() -> None:
    args = _parse_args()
    root = args.round1.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    starts = _csv(root / "direct_fit/direct_fit_starts.csv")
    reference = _csv(root / "direct_fit/direct_fit_reference.csv")
    information = _csv(root / "screen/window_information_scores.csv")
    states = _csv(root / "screen/state_feature_rows.csv")
    selection = json.loads((root / "screen/selected_objectives.json").read_text())
    screen = json.loads((root / "screen/screen_manifest.json").read_text())
    floors = json.loads((root / "screen/noise_floors.json").read_text())

    ranking = selection["ranking"]
    winner = selection["raw"]
    truth = next(row for row in states if row["name"] == "reference_truth")
    selected_keys = {
        "Fine EGI onset CVaR90": "egi_1.4mm__onset__cvar90",
        "Middle EGI late RMS": "egi_5.8mm__late__rms",
        "Broad EGI onset CVaR90": "egi_11.4mm__onset__cvar90",
        "FRE late coherent RMS": "fre__late__coherent_rms",
    }
    ratios = {label: float(truth[key]) / float(floors[key]) for label, key in selected_keys.items()}
    winner_info = [row for row in information if row["candidate"] == winner["name"]]

    with PdfPages(output) as pdf:
        _summary_page(pdf, screen, winner, ranking, ratios)
        _direct_fit_page(pdf, starts, reference)
        _objective_page(pdf, ranking, winner_info)
        _regime_noise_page(pdf, screen, ratios)
        _decision_page(pdf)
    print(f"round1 audit report={output}", flush=True)


def _summary_page(pdf, screen, winner, ranking, ratios):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle("Notched-EBW Round 1: independent audit", fontsize=19, y=.95)
    verdict = (
        "Verdict: Round 1 is informative but does not yet justify freezing the "
        "automatic objective unchanged. Advance the 7/29/57 raw formulation to a "
        "small online pilot only after repairing the temporal regimes; keep the "
        "7/57 formulation as the parsimonious comparator."
    )
    facts = [
        f"Evidence: {screen['states']} states, {screen['noise_replicates']} WDBN1 noise realisations, four physical EGI supports.",
        f"Automatic winner: {winner['name']} (alpha={winner['alpha']}); full candidate merit range is only {ranking[0]['merit']-ranking[-1]['merit']:.4f}.",
        "Direct representability reaches 3.26 MPa ROI RMSE at BF7; BF8 adds no improvement.",
        "Phase-0 yielded fraction peaks at 16.25%, below the configured 20% developed and 65% late thresholds.",
        "The nonempty fallback therefore assigns frame 13 to both developed and late; these are duplicated one-frame blocks.",
        "High-plastic within-BF discrimination weakens sharply by BF7-BF8 (rho about 0.10 to 0.07 for the winner).",
        "Three of four selected truth residual features are at or below their estimated noise floor; FRE is only 3% above it.",
    ]
    fig.text(.065, .84, "\n".join(textwrap.fill(line, 105) for line in [verdict, *facts]), va="top", fontsize=11.2, linespacing=1.55)
    fig.text(.065, .18, "Recommendation", fontsize=14, weight="bold")
    fig.text(.065, .135, "Repair regime resolution using data-relative/quantile thresholds, rerun the cheap offline scoring, then run a matched two-seed clean/noisy pilot of controls + raw hybrids. Do not launch the full Round 2 factorial or projected objective yet.", fontsize=11.5, wrap=True)
    pdf.savefig(fig); plt.close(fig)


def _direct_fit_page(pdf, starts, reference):
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), constrained_layout=True)
    bfs = sorted({int(row["basis_count"]) for row in starts})
    grouped = [[float(row["roi_rmse_mpa"]) for row in starts if int(row["basis_count"]) == bf] for bf in bfs]
    axes[0].boxplot(grouped, tick_labels=bfs, showfliers=True)
    axes[0].plot(np.arange(1, len(bfs)+1), [min(values) for values in grouped], "o-", color="tab:red", label="best start")
    axes[0].set(xlabel="SPD Gaussian BF count", ylabel="ROI RMSE [MPa]", title="Direct-fit multistart spread")
    axes[0].grid(alpha=.25); axes[0].legend()
    rbfs = [int(row["basis_count"]) for row in reference]
    for key, label in (("roi_rmse_mpa", "ROI"), ("yielded_rmse_mpa", "Yielded"), ("high_plastic_rmse_mpa", "High plastic")):
        axes[1].plot(rbfs, [float(row[key]) for row in reference], marker="o", label=label)
    axes[1].set(xlabel="SPD Gaussian BF count", ylabel="Best RMSE [MPa]", title="Best representability curve")
    axes[1].grid(alpha=.25); axes[1].legend()
    fig.suptitle("Direct fitting: excellent capacity, increasing nonconvexity", fontsize=16)
    fig.text(.5, .015, "BF0-BF2 converge consistently. From BF3 the starts spread; BF4-BF8 hit the iteration limit. The best sequential start remains strong, but the curve is a conservative reference rather than a certified global optimum.", ha="center", fontsize=9.5)
    pdf.savefig(fig); plt.close(fig)


def _objective_page(pdf, ranking, winner_info):
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), constrained_layout=True)
    names = [row["name"].replace("raw_", "") for row in ranking]
    merits = [float(row["merit"]) for row in ranking]
    axes[0].barh(np.arange(len(names)), merits)
    axes[0].set_yticks(np.arange(len(names)), names); axes[0].invert_yaxis()
    axes[0].set_xlim(min(merits)-.003, max(merits)+.001)
    axes[0].set(xlabel="Development/validation mean Spearman", title="Candidates are nearly tied")
    axes[0].grid(axis="x", alpha=.25)
    bf_rows = [row for row in winner_info if row["subset"].startswith("diagnostic_bf")]
    for target, label in (("yielded_rmse_mpa", "Yielded"), ("high_plastic_rmse_mpa", "High plastic")):
        selected = sorted([row for row in bf_rows if row["target"] == target], key=lambda row: int(row["subset"].split("bf")[1]))
        axes[1].plot([int(row["subset"].split("bf")[1]) for row in selected], [float(row["spearman_r"]) for row in selected], marker="o", label=label)
    axes[1].axhline(0.75, color="grey", linestyle="--", linewidth=1, label="0.75 reference")
    axes[1].set(xlabel="BF count", ylabel="Within-BF Spearman rho", ylim=(0, 1), title="Winner on optimiser states")
    axes[1].grid(alpha=.25); axes[1].legend()
    fig.suptitle("Objective evidence: strong broad ranking, weak late local discrimination", fontsize=16)
    pdf.savefig(fig); plt.close(fig)


def _regime_noise_page(pdf, screen, ratios):
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), constrained_layout=True)
    fraction = np.asarray(screen["load_regimes"]["yielded_fraction"], dtype=float)
    axes[0].plot(np.arange(fraction.size), 100*fraction, marker="o")
    for value, label in ((2, "onset"), (20, "developed"), (65, "late")):
        axes[0].axhline(value, linestyle="--", label=f"{label} {value}%")
    axes[0].set(xlabel="Frame", ylabel="Phase-0 yielded fraction [%]", title="Configured regimes do not fit this test")
    axes[0].grid(alpha=.25); axes[0].legend()
    labels = list(ratios); values = [ratios[label] for label in labels]
    axes[1].barh(np.arange(len(labels)), values, color=["tab:red" if value <= 1 else "tab:orange" for value in values])
    axes[1].axvline(1.0, color="black", linewidth=1, label="truth = noise floor")
    axes[1].set_yticks(np.arange(len(labels)), labels); axes[1].invert_yaxis()
    axes[1].set(xlabel="Truth residual / WDBN1 noise floor", title="Selected-feature signal margin")
    axes[1].grid(axis="x", alpha=.25); axes[1].legend()
    fig.suptitle("Temporal and noise audit", fontsize=16)
    pdf.savefig(fig); plt.close(fig)


def _decision_page(pdf):
    fig = plt.figure(figsize=(11.69, 8.27)); fig.suptitle("Recommended next steps", fontsize=18, y=.94)
    sections = [
        ("1. Repair and rerun the offline aggregation (required)", "Replace absolute 20%/65% thresholds and duplicate-frame fallback with monotone, data-relative regimes that guarantee disjoint multi-frame blocks. Re-score existing state residuals where possible; regenerate only metrics that depend on changed frame membership."),
        ("2. Retain two raw formulations", "Carry 7/29/57 as the information-rich candidate and 7/57 as the parsimonious candidate. Alpha=0.25 and 0.5 are effectively tied; test both in the small pilot rather than declaring 0.25 uniquely optimal."),
        ("3. Run a small online pilot", "After the regime fix, run current 29/57, multiscale equal, 7/57 raw, and 7/29/57 raw with two matched seeds under clean and 1x noise. Cap at BF6 or BF7 and compare recovery-gap AUC, yielded/high-plastic errors and closure."),
        ("4. Projection remains gated", "Complete online native-DOF projection preparation, BF-refresh/restoration tests and conditioning diagnostics before including the projected hybrid."),
        ("5. Direct-fit follow-up is narrow", "Do not repeat a general BF7-vs-BF8 campaign. Optionally continue only the best BF6-BF8 direct-fit states with a larger iteration budget to confirm the 3.26 MPa plateau."),
    ]
    y=.85
    for heading, body in sections:
        fig.text(.07,y,heading,fontsize=13,weight="bold"); y-=.045
        fig.text(.085,y,textwrap.fill(body,118),fontsize=10.2,va="top"); y-=.115
    fig.text(.07,.025,"Decision: do not start the full 16-case Round 2 matrix yet. Repair the regimes, then run the targeted pilot.",fontsize=10.8,weight="bold",wrap=True)
    pdf.savefig(fig); plt.close(fig)


def _csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round1",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
