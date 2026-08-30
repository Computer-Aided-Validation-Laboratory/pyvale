"""Create a concise PDF report for the notched-EBW offline selector study."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import textwrap

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


INK = "#17202a"
MUTED = "#52606d"
BLUE = "#1565c0"
GREEN = "#2e7d32"
ORANGE = "#ef6c00"
RED = "#c62828"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rankings = _read(args.results / "candidate_rankings.csv", "candidate")
    selections = _read(args.results / "trajectory_selections.csv", "candidate")
    force = np.load(args.dataset / "prepared/force.npy")[:, 0]
    time = np.load(args.dataset / "prepared/time.npy")
    plastic_fraction = np.asarray(
        [0.0, 0.0, 0.0, 0.0010, 0.0252, 0.0730, 0.1064,
         0.1340, 0.1561, 0.1777, 0.2116, 0.2482, 0.2790, 0.2958]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now().astimezone()
    with PdfPages(
        args.output,
        metadata={
            "Title": "Notched-EBW offline scalar-selector study",
            "Author": "PyVale investigation",
            "CreationDate": generated,
        },
    ) as pdf:
        _executive_page(pdf, rankings, selections, generated)
        _candidate_page(pdf, rankings, selections)
        _temporal_page(pdf, rankings, time, force, plastic_fraction)
        _next_steps_page(pdf)
    print(args.output)


def _read(path: Path, key: str):
    with path.open(newline="", encoding="utf-8") as stream:
        return {row[key]: row for row in csv.DictReader(stream)}


def _page():
    figure = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    axis = figure.add_axes((0.06, 0.06, 0.88, 0.88))
    axis.axis("off")
    return figure, axis


def _title(axis, title, subtitle=None):
    axis.text(0.0, 1.0, title, fontsize=22, fontweight="bold", color=INK, va="top")
    if subtitle:
        axis.text(0.0, 0.945, subtitle, fontsize=10.5, color=MUTED, va="top")


def _wrapped(axis, x, y, text, width=105, **kwargs):
    axis.text(x, y, textwrap.fill(text, width=width), **kwargs)


def _value(rows, candidate, field):
    return float(rows[candidate][field])


def _executive_page(pdf, rankings, selections, generated):
    figure, axis = _page()
    _title(
        axis,
        "Offline EGI/FRE scalar-selector study",
        f"150 stored synthetic states | 60 scalar candidates | {generated:%d %B %Y, %H:%M %Z}",
    )
    axis.text(
        0.0, 0.855, "Outcome", fontsize=12, fontweight="bold", color="white",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": BLUE, "edgecolor": "none"},
    )
    _wrapped(
        axis, 0.0, 0.79,
        "Do not replace the production objective with any tested scalar yet. Delaying aggregation reveals useful information—especially around first yield—but no candidate materially improves model-order selection or the ~39 MPa yielded-error floor.",
        fontsize=14, fontweight="bold", color=INK, va="top",
    )

    current = rankings["current"]
    best_global = rankings["time_p90__max_metric"]
    onset = rankings["load_step_03"]
    best_selection = selections["tail_p95__equal_sum"]
    rows = [
        ["Current J", f"{float(current['campaign_yielded_rmse_spearman']):.3f}", f"{float(current['campaign_late_yielded_rmse_spearman']):.3f}", f"{float(current['final_yielded_rmse_spearman']):.3f}", f"{float(current['campaign_late_high_plastic_rmse_spearman']):.3f}", "39.0 MPa"],
        ["Time-P90, max metric", f"{float(best_global['campaign_yielded_rmse_spearman']):.3f}", f"{float(best_global['campaign_late_yielded_rmse_spearman']):.3f}", f"{float(best_global['final_yielded_rmse_spearman']):.3f}", f"{float(best_global['campaign_late_high_plastic_rmse_spearman']):.3f}", "39.0 MPa"],
        ["Yield-onset step 3", f"{float(onset['campaign_yielded_rmse_spearman']):.3f}", f"{float(onset['campaign_late_yielded_rmse_spearman']):.3f}", f"{float(onset['final_yielded_rmse_spearman']):.3f}", f"{float(onset['campaign_late_high_plastic_rmse_spearman']):.3f}", "52.4 MPa"],
        ["Best minimum-score selector", "—", "—", "—", "—", f"{float(best_selection['median_yielded_rmse_mpa']):.1f} MPa"],
    ]
    table = axis.table(
        cellText=rows,
        colLabels=["Scalar", "All-state ρ", "BF5–8 ρ", "Final ρ", "BF5–8 high-plastic ρ", "Selected RMSE"],
        cellLoc="center", colLoc="center", bbox=(0.0, 0.42, 1.0, 0.25),
        colWidths=[0.22, 0.145, 0.145, 0.145, 0.20, 0.145],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.3)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e2ec")
        if row == 0:
            cell.set_facecolor("#eaf2f8")
            cell.set_text_props(weight="bold", color=INK)

    findings = [
        "Time-P90/max-metric aggregation improves gross ranking from ρ=0.845 to 0.881, but still selects BF8 and has no useful final-endpoint ranking.",
        "The best minimum-score selector changes one of eight trajectories and improves median yielded RMSE by only 0.10 MPa—far below a meaningful gain.",
        "Broad-scale EGI-57 is consistently strongest for late yielded-map ranking; coherent FRE is the only tested component with positive late high-plastic discrimination.",
        "The duplicated 0%/0.5% trajectories and exploratory multiple comparisons mean all candidate findings require independent validation.",
    ]
    axis.text(0.0, 0.36, "Decision evidence", fontsize=13, fontweight="bold", color=INK)
    y = 0.31
    for finding in findings:
        axis.text(0.008, y, "•", fontsize=15, color=BLUE, va="top")
        _wrapped(axis, 0.035, y, finding, fontsize=10.7, color=INK, va="top")
        y -= 0.074
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def _candidate_page(pdf, rankings, selections):
    figure, axis = _page()
    _title(axis, "What survives delayed aggregation?")
    candidates = [
        ("current", "Current weighted RMS"),
        ("time_p90__max_metric", "Time-P90 / max metric"),
        ("plastic_active_rms__egi57", "Plastic-active EGI-57"),
        ("tail_p95__egi57", "P95-tail EGI-57"),
        ("coherent_rms__fre", "Coherent FRE"),
        ("load_step_03", "Yield-onset step 3"),
    ]
    columns = ["Candidate", "All yielded ρ", "Late yielded ρ", "Final yielded ρ", "Late high-plastic ρ", "Controlled yielded ρ", "Late pair accuracy"]
    values = []
    for key, label in candidates:
        row = rankings[key]
        values.append([
            label,
            f"{float(row['campaign_yielded_rmse_spearman']):.3f}",
            f"{float(row['campaign_late_yielded_rmse_spearman']):.3f}",
            f"{float(row['final_yielded_rmse_spearman']):.3f}",
            f"{float(row['campaign_late_high_plastic_rmse_spearman']):.3f}",
            f"{float(row['controlled_yielded_rmse_spearman']):.3f}",
            f"{float(row['late_adjacent_yielded_pairwise']):.3f}",
        ])
    table = axis.table(
        cellText=values, colLabels=columns, cellLoc="center", colLoc="center",
        bbox=(0.0, 0.50, 1.0, 0.36),
        colWidths=[0.21, 0.13, 0.13, 0.13, 0.15, 0.14, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.9)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e2ec")
        if row == 0:
            cell.set_facecolor("#eaf2f8")
            cell.set_text_props(weight="bold", color=INK)
        elif column == 0:
            cell.set_text_props(ha="left")

    axis.text(0.0, 0.425, "Interpretation", fontsize=13, fontweight="bold", color=INK)
    paragraphs = [
        "No single ordering is robust across gross progress, plausible late maps, highly plastic material, and independent controlled perturbations. This is exactly the information loss expected from collapsing heterogeneous evidence too early.",
        "The strongest components are complementary: EGI-57 contains late yield-map information, while spatially coherent FRE contains different information about the high-plastic tail. A fixed weighted average can allow one to compensate for deterioration in the other.",
        f"Absolute minimisation remains unsafe: even the best tested selector ({min(selections, key=lambda key: float(selections[key]['median_yielded_rmse_mpa']))}) produces {min(float(row['median_yielded_rmse_mpa']) for row in selections.values()):.2f} MPa median yielded RMSE.",
    ]
    y = 0.37
    for paragraph in paragraphs:
        _wrapped(axis, 0.0, y, paragraph, fontsize=11.2, color=INK, va="top")
        y -= 0.105
    axis.text(
        0.0, 0.055,
        "Conclusion: preserve a vector of metric × load-regime evidence through optimisation and collapse only at the acceptance/decision layer.",
        fontsize=12.5, fontweight="bold", color=BLUE, va="top",
    )
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def _temporal_page(pdf, rankings, time, force, plastic_fraction):
    figure = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    figure.suptitle("Temporal structure: yield onset is informative", x=0.06, y=0.96, ha="left", fontsize=22, fontweight="bold", color=INK)
    axis = figure.add_axes((0.08, 0.34, 0.84, 0.48))
    steps = np.arange(len(time))
    late_yield = np.asarray([_value(rankings, f"load_step_{i:02d}", "campaign_late_yielded_rmse_spearman") for i in steps])
    final_yield = np.asarray([_value(rankings, f"load_step_{i:02d}", "final_yielded_rmse_spearman") for i in steps])
    axis.plot(steps, late_yield, "o-", lw=2, color=BLUE, label="BF5–8 yielded correlation")
    axis.plot(steps, final_yield, "s--", lw=1.8, color=ORANGE, label="Final-endpoint yielded correlation")
    axis.axvline(3, color=RED, ls=":", lw=2, label="first yielding")
    axis.axhline(0.0, color=MUTED, lw=0.8)
    axis.set(xlabel="Load-step index", ylabel="Spearman ρ", xticks=steps, ylim=(-1.0, 1.0))
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, loc="lower right")
    force_axis = axis.twinx()
    force_axis.plot(steps, force / 1000.0, color=GREEN, alpha=0.5, lw=1.5, label="force")
    force_axis.set_ylabel("Force [kN]", color=GREEN)

    figure.text(
        0.08, 0.255,
        f"Step 3: t={time[3]:.4f}, force={force[3]/1000.0:.2f} kN, only {100*plastic_fraction[3]:.1f}% of ROI plastic. "
        f"Late yielded ρ={late_yield[3]:.3f}; final yielded ρ={final_yield[3]:.3f}.",
        fontsize=12, fontweight="bold", color=RED,
    )
    figure.text(
        0.08, 0.17,
        textwrap.fill(
            "Force-squared temporal weighting favours the highest loads, where signal-to-noise is good but yield-strength and hardening effects are more entangled. The onset window can be more discriminative of yield strength. However, step 3 has controlled-state ρ≈0 and its absolute minimum selects poor early models, so the result must be treated as a window-design hypothesis, not a new objective.",
            width=130,
        ),
        fontsize=11.2, color=INK, va="top",
    )
    figure.text(
        0.08, 0.055,
        "Test next: pre-yield calibration, yield-onset, developed-plasticity, and late-plasticity windows kept separate through validation.",
        fontsize=12.5, fontweight="bold", color=BLUE,
    )
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def _next_steps_page(pdf):
    figure, axis = _page()
    _title(axis, "Next implementation plan")
    steps = [
        ("1  Expand the independent synthetic state library", "Add local weld/HAZ perturbations, width/centre/amplitude changes, high-plastic-tail errors, hardening compensation, and maps not generated by the optimiser. Split development and validation states before choosing weights or thresholds."),
        ("2  Introduce named load regimes", "Keep EGI-29, EGI-57, and FRE summaries separately for pre-yield (0–2), onset (3–5), developed plasticity (6–9), and late plasticity (10–13). Record RMS, tail, and coherence per regime; do not combine them inside the metric layer."),
        ("3  Build a guarded growth selector", "Continue using current J for optimisation initially. Accept a proposed BF only if training J improves, held-out onset/developed-plasticity evidence improves, coherent FRE does not materially regress, and the new BF/Jacobian direction is sufficiently novel and well-conditioned."),
        ("4  Validate without leakage", "Tune on development maps, then freeze the selector. Test rank correlation, pairwise accuracy, chosen model order, yielded/high-plastic RMSE, and noise stability on unseen maps and load/noise variants."),
        ("5  Run the next optimiser campaign", "Only after offline validation: sensitivity-SPD, 8 independent seeds, temporary cap 7 BFs, concise progress output. Compare against the existing 0% trajectories; defer the queued 5% and EGI runs."),
    ]
    y = 0.87
    for index, (heading, body) in enumerate(steps):
        color = BLUE if index < 3 else GREEN
        axis.text(0.0, y, heading, fontsize=13, fontweight="bold", color=color, va="top")
        _wrapped(axis, 0.025, y - 0.045, body, fontsize=10.8, color=INK, va="top")
        y -= 0.151
    axis.text(0.0, 0.10, "Go/no-go evidence for a replacement scalar or selector", fontsize=12.5, fontweight="bold", color=INK)
    _wrapped(
        axis, 0.025, 0.055,
        "Require independent late/final yielded ρ and pairwise accuracy ≥0.70, positive high-plastic discrimination, and a material selected-map improvement (target ≥5 MPa rather than the observed 0.10 MPa), with stable behaviour under realistic noise. Until then, retain the existing EGI/FRE objective and treat alternatives as diagnostics.",
        fontsize=10.8, color=INK, va="top",
    )
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
