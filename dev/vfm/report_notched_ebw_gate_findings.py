"""Create a concise decision report from a notched-EBW gate campaign."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
import os
from pathlib import Path
from statistics import median
import textwrap

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


CAMPAIGN_SOURCE = "campaign"
ACCENT = "#1565c0"
ORANGE = "#ef6c00"
GREEN = "#2e7d32"
RED = "#c62828"
INK = "#17202a"
MUTED = "#52606d"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = _load_rows(args.analysis / "state_metrics.csv")
    accepted = [
        row
        for row in rows
        if row["source"] == CAMPAIGN_SOURCE and row["accepted"]
    ]
    final = [row for row in accepted if row["is_final_accepted"]]
    truth = next(row for row in rows if row["state_id"] == "controlled/truth")
    if not accepted or not final:
        raise ValueError("The analysis contains no accepted campaign states.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now().astimezone()
    with PdfPages(
        args.output,
        metadata={
            "Title": "Notched-EBW gate campaign: findings and next steps",
            "Author": "PyVale investigation",
            "CreationDate": generated,
        },
    ) as pdf:
        _executive_page(pdf, accepted, final, truth, generated)
        _trajectory_page(pdf, accepted)
        _selection_page(pdf, accepted, final, truth)
        _next_steps_page(pdf)
    print(args.output)


def _load_rows(path: Path) -> list[dict[str, object]]:
    float_fields = {
        "gate",
        "objective",
        "active_objective",
        "roi_rmse_mpa",
        "yielded_rmse_mpa",
        "high_plastic_rmse_mpa",
        "yielded_mape_percent",
        "yielded_above_5pct",
        "yielded_above_10pct",
        "yielded_above_15pct",
        "hardening_error_percent",
    }
    int_fields = {"seed", "solve_index", "basis_count"}
    bool_fields = {"accepted", "is_final_accepted", "is_best_visited"}
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            row: dict[str, object] = dict(raw)
            for key in float_fields:
                row[key] = float(raw[key]) if raw[key] else np.nan
            for key in int_fields:
                row[key] = int(raw[key]) if raw[key] else -1
            for key in bool_fields:
                row[key] = raw[key] == "True"
            rows.append(row)
    return rows


def _groups(rows: list[dict[str, object]]) -> dict[float, list[dict[str, object]]]:
    groups: dict[float, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[float(row["gate"])].append(row)
    return dict(sorted(groups.items()))


def _gate_label(gate: float) -> str:
    return f"{100.0 * gate:g}%"


def _page() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    axis = fig.add_axes((0.06, 0.06, 0.88, 0.88))
    axis.axis("off")
    return fig, axis


def _title(axis: plt.Axes, title: str, subtitle: str | None = None) -> None:
    axis.text(0.0, 1.0, title, fontsize=22, fontweight="bold", color=INK, va="top")
    if subtitle:
        axis.text(0.0, 0.945, subtitle, fontsize=10.5, color=MUTED, va="top")


def _wrapped(axis: plt.Axes, x: float, y: float, text: str, **kwargs) -> None:
    width = kwargs.pop("width", 90)
    axis.text(x, y, textwrap.fill(text, width=width), **kwargs)


def _executive_page(pdf, accepted, final, truth, generated) -> None:
    fig, axis = _page()
    _title(
        axis,
        "Notched-EBW gate campaign: decision report",
        f"15 completed runs; 119 campaign solve states | {generated:%d %B %Y, %H:%M %Z}",
    )
    axis.text(
        0.0,
        0.86,
        "Decision",
        fontsize=12,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": ACCENT, "edgecolor": "none"},
    )
    _wrapped(
        axis,
        0.0,
        0.79,
        "Use seven basis functions as the temporary investigation cap. Do not resume the queued 5% gate or EGI-control jobs yet. The 0.5% gate was effectively inactive and BF8 usually reduced mechanical cost while degrading map recovery.",
        fontsize=14,
        fontweight="bold",
        color=INK,
        va="top",
        width=105,
    )

    columns = ["Gate", "Runs", "Median J", "ROI RMSE", "Yielded RMSE", "High-plastic RMSE", ">10% error", ">15% error"]
    cell_text = []
    for gate, group in _groups(final).items():
        cell_text.append(
            [
                _gate_label(gate),
                str(len(group)),
                f"{median(float(r['objective']) for r in group):.5f}",
                f"{median(float(r['roi_rmse_mpa']) for r in group):.1f} MPa",
                f"{median(float(r['yielded_rmse_mpa']) for r in group):.1f} MPa",
                f"{median(float(r['high_plastic_rmse_mpa']) for r in group):.1f} MPa",
                f"{100 * median(float(r['yielded_above_10pct']) for r in group):.1f}%",
                f"{100 * median(float(r['yielded_above_15pct']) for r in group):.1f}%",
            ]
        )
    table = axis.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=(0.0, 0.48, 1.0, 0.19),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e2ec")
        if row == 0:
            cell.set_facecolor("#eaf2f8")
            cell.set_text_props(weight="bold", color=INK)

    bullets = [
        "The two gates are not independent repeats: six of seven paired final maps are identical. A 0.5% threshold cannot discriminate candidates whose late objective gains are about 1.5–2.5%.",
        "Both policies reached the imposed maximum of 8 BFs. Their yielded-region endpoint errors are effectively equal (39.0 versus 38.9 MPa).",
        f"The median final J is below the known-map J={float(truth['objective']):.5f}, yet the yielded RMSE remains about 39 MPa. This confirms that lower in-sample closure cost is not sufficient evidence of a better yield map.",
        "The most plastically active material remains worst recovered: median high-plastic RMSE is 56–57 MPa.",
    ]
    axis.text(0.0, 0.415, "What this campaign establishes", fontsize=13, fontweight="bold", color=INK)
    y = 0.365
    for bullet in bullets:
        axis.text(0.008, y, "•", fontsize=15, color=ACCENT, va="top")
        _wrapped(axis, 0.035, y, bullet, fontsize=10.7, color=INK, va="top", width=110)
        y -= 0.087
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _trajectory_page(pdf, accepted) -> None:
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    fig.suptitle("Model-order evidence", x=0.06, y=0.96, ha="left", fontsize=22, fontweight="bold", color=INK)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.83, bottom=0.29, wspace=0.24)
    colors = [ACCENT, ORANGE]
    by_gate = _groups(accepted)
    for color, (gate, group) in zip(colors, by_gate.items(), strict=False):
        by_basis: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in group:
            by_basis[int(row["basis_count"])].append(row)
        bases = sorted(by_basis)
        objectives = [median(float(r["objective"]) for r in by_basis[b]) for b in bases]
        yielded = [median(float(r["yielded_rmse_mpa"]) for r in by_basis[b]) for b in bases]
        axes[0].plot(bases, objectives, "o-", color=color, lw=2, label=f"gate {_gate_label(gate)}")
        axes[1].plot(bases, yielded, "o-", color=color, lw=2, label=f"gate {_gate_label(gate)}")
    axes[0].set(title="Mechanical objective", xlabel="Accepted basis count", ylabel="Median J")
    axes[1].set(title="Yield-map recovery", xlabel="Accepted basis count", ylabel="Median yielded RMSE [MPa]")
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
        ax.set_xticks(range(1, 9))

    increments = _late_increment_summary(accepted)
    summary = (
        f"BF6: yielded RMSE improved in {increments[6][2]}/{increments[6][3]} pairs; "
        f"median change {increments[6][1]:+.2f} MPa.\n"
        f"BF7: improved in {increments[7][2]}/{increments[7][3]} pairs; "
        f"median change {increments[7][1]:+.2f} MPa.\n"
        f"BF8: improved in only {increments[8][2]}/{increments[8][3]} pairs; "
        f"median change {increments[8][1]:+.2f} MPa, despite median J falling {increments[8][0]:.2f}%."
    )
    fig.text(0.08, 0.235, "Paired late-growth result", fontsize=12, fontweight="bold", color=INK, va="top")
    fig.text(0.08, 0.195, summary, fontsize=11.2, color=INK, va="top")
    fig.text(
        0.08,
        0.055,
        "Conclusion: BF7 is useful; BF8 is the first consistent sign of objective-driven overfitting.",
        fontsize=11.5,
        fontweight="bold",
        color=RED,
        ha="left",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _late_increment_summary(accepted):
    increments: dict[int, list[tuple[float, float]]] = defaultdict(list)
    cases: dict[tuple[float, int], list[dict[str, object]]] = defaultdict(list)
    for row in accepted:
        cases[(float(row["gate"]), int(row["seed"]))].append(row)
    for rows in cases.values():
        ordered = sorted(rows, key=lambda row: int(row["basis_count"]))
        for before, after in zip(ordered, ordered[1:]):
            basis = int(after["basis_count"])
            if basis not in (6, 7, 8):
                continue
            j_improvement = 100.0 * (
                float(before["objective"]) - float(after["objective"])
            ) / float(before["objective"])
            rmse_change = float(after["yielded_rmse_mpa"]) - float(before["yielded_rmse_mpa"])
            increments[basis].append((j_improvement, rmse_change))
    return {
        basis: (
            median(value[0] for value in values),
            median(value[1] for value in values),
            sum(value[1] < 0.0 for value in values),
            len(values),
        )
        for basis, values in increments.items()
    }


def _selection_page(pdf, accepted, final, truth) -> None:
    fig, axis = _page()
    _title(axis, "Why the current acceptance rule is not robust")
    rank_points = [
        ("All 119 campaign states", "J vs yielded RMSE", "Spearman ρ = 0.845", GREEN),
        ("15 final endpoints", "J vs yielded RMSE", "ρ = 0.505; p = 0.055", ORANGE),
        ("15 final endpoints", "J vs high-plastic RMSE", "ρ = −0.563; p = 0.029", RED),
        ("15 final endpoints", "active score vs yielded RMSE", "ρ = 0.332; p = 0.226", RED),
    ]
    y = 0.84
    for scope, comparison, result, color in rank_points:
        axis.add_patch(plt.Rectangle((0.0, y - 0.035), 0.018, 0.055, color=color, transform=axis.transAxes))
        axis.text(0.035, y, scope, fontsize=11, fontweight="bold", color=INK, va="center")
        axis.text(0.30, y, comparison, fontsize=10.5, color=MUTED, va="center")
        axis.text(0.72, y, result, fontsize=11, fontweight="bold", color=color, va="center")
        y -= 0.095

    _wrapped(
        axis,
        0.0,
        0.46,
        "J tracks gross progress from one to several basis functions, but loses reliable discrimination among the plausible low-cost endpoints. At that point, lower J can correspond to worse recovery in the highly plastic region. The proposed sensitivity-active diagnostic changes the correlations only marginally and is not ready to replace J.",
        fontsize=12,
        color=INK,
        va="top",
        width=105,
    )
    axis.text(0.0, 0.29, "Implication for the gate", fontsize=13, fontweight="bold", color=INK)
    replay = _replay_scalar_gates(accepted)
    _wrapped(
        axis,
        0.0,
        0.235,
        f"A scalar percentage gate only asks whether training J falls. Offline replay gives median yielded RMSE of {replay[0.005]:.1f}, {replay[0.01]:.1f}, {replay[0.02]:.1f}, and {replay[0.05]:.1f} MPa for 0.5%, 1%, 2%, and 5% gates: none breaks the ~39 MPa floor. The known map has J={float(truth['objective']):.5f}, while final models reach lower J with substantial map error.",
        fontsize=11.5,
        color=INK,
        va="top",
        width=105,
    )
    axis.text(
        0.0,
        0.09,
        "Required change: accept model growth using independent evidence—held-out active load windows and basis/Jacobian novelty—alongside J.",
        fontsize=13,
        fontweight="bold",
        color=ACCENT,
        va="top",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _replay_scalar_gates(accepted):
    cases: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in accepted:
        if float(row["gate"]) == 0.0:
            cases[int(row["seed"])].append(row)
    results = {}
    for threshold in (0.005, 0.01, 0.02, 0.05):
        selected = []
        for rows in cases.values():
            ordered = sorted(rows, key=lambda row: int(row["basis_count"]))
            current = ordered[0]
            for candidate in ordered[1:]:
                gain = (
                    float(current["objective"]) - float(candidate["objective"])
                ) / float(current["objective"])
                if gain < threshold:
                    break
                current = candidate
            selected.append(current)
        results[threshold] = median(
            float(row["yielded_rmse_mpa"]) for row in selected
        )
    return results


def _next_steps_page(pdf) -> None:
    fig, axis = _page()
    _title(axis, "Recommended route to experimental-data readiness")
    steps = [
        (
            "1  Offline selector — first, no new solves",
            "Replay every stored basis trajectory. Compare stopping rules using training-J gain, predicted-plastic/active-window validation residual, BF novelty, and conditioning. Select thresholds against synthetic yielded/high-plastic RMSE. This extracts more value from the 119 states immediately.",
            ACCENT,
        ),
        (
            "2  Add independent acceptance evidence",
            "Optimise on designated load/equilibrium windows, but accept a new BF only when held-out active windows also improve. Require the proposed BF response/Jacobian column to retain sufficient norm after projection onto the existing basis span; reject redundant or ill-conditioned growth.",
            ACCENT,
        ),
        (
            "3  Focused confirmation campaign",
            "Run sensitivity-SPD only, 8 seeds, cap 7 BFs, with the new selector and concise 60 s aggregate progress. Compare against the existing trajectories. Do not spend time on the queued 5% gate until this rule is tested.",
            GREEN,
        ),
        (
            "4  Stress-test before experiment",
            "Repeat the winning rule with synthetic strain noise, force noise/bias, load-window omissions, and modest model mismatch. Freeze hyperparameters before looking at experimental results; report seed spread and spatial error in the plastic zone, not J alone.",
            GREEN,
        ),
    ]
    y = 0.87
    for heading, body, color in steps:
        axis.text(0.0, y, heading, fontsize=13, fontweight="bold", color=color, va="top")
        _wrapped(axis, 0.025, y - 0.048, body, fontsize=10.8, color=INK, va="top", width=105)
        y -= 0.195
    axis.text(0.0, 0.105, "Proposed synthetic release criteria", fontsize=12.5, fontweight="bold", color=INK)
    _wrapped(
        axis,
        0.025,
        0.058,
        "Predeclare thresholds before running: median yielded RMSE ≤20 MPa, median high-plastic RMSE ≤30 MPa, no material tail degradation from the final accepted BF, and stable selection across seeds/noise cases. These are ambitious but credible relative to the ~13 MPa fixed-geometry result.",
        fontsize=10.8,
        color=INK,
        va="top",
        width=110,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
