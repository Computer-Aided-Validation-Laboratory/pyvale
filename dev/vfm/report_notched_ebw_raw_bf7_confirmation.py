"""Create a concise decision report for the repaired raw BF7 confirmation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import textwrap

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from report_notched_ebw_raw_pilot_findings import (
    LABELS,
    Row,
    _csv,
    _load_rows,
    _write_rows,
)


RAW_OBJECTIVES = ("raw_parsimonious", "raw_information_rich")
CONTROL_OBJECTIVES = ("current_29_57", "multiscale_equal_7_29_57")
COLORS = {
    "current_29_57": "0.55",
    "multiscale_equal_7_29_57": "tab:blue",
    "raw_parsimonious": "tab:orange",
    "raw_information_rich": "tab:green",
}
METRICS = (
    ("yielded_rmse_mpa", "Yielded RMSE"),
    ("high_plastic_rmse_mpa", "High-plastic RMSE"),
    ("roi_rmse_mpa", "Whole-ROI RMSE"),
)


def main() -> None:
    args = _parse_args()
    dataset = args.dataset.expanduser().resolve()
    campaign = args.campaign.expanduser().resolve()
    controls = args.controls.expanduser().resolve()
    direct = _csv(args.direct_fit.expanduser().resolve())
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    raw_rows = _load_rows(campaign, dataset)
    control_rows = [
        row
        for row in _load_rows(controls, dataset)
        if row.objective in CONTROL_OBJECTIVES
    ]
    rows = raw_rows + control_rows
    _validate(raw_rows)
    _write_rows(output.with_suffix(".csv"), raw_rows)
    summary = _build_summary(raw_rows, control_rows)
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    output.with_suffix(".md").write_text(
        _markdown(summary),
        encoding="utf-8",
    )

    with PdfPages(output) as pdf:
        _executive_page(pdf, summary)
        _trajectory_page(pdf, rows, direct)
        _decision_page(pdf, summary)
    print(f"raw BF7 confirmation findings={output}", flush=True)


def _validate(rows: list[Row]) -> None:
    expected = {
        (objective, condition, seed, basis_count)
        for objective in RAW_OBJECTIVES
        for condition in ("clean", "noise1x")
        for seed in (0, 1)
        for basis_count in range(1, 8)
    }
    observed = {
        (row.objective, row.condition, row.seed, row.basis_count)
        for row in rows
    }
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"Expected the complete 56-state raw BF1-BF7 matrix; "
            f"missing={missing[:5]} extra={extra[:5]}."
        )
    if not all(row.accepted for row in rows):
        raise ValueError("Fixed-trajectory confirmation contains rejected states.")


def _build_summary(
    raw_rows: list[Row],
    control_rows: list[Row],
) -> dict[str, object]:
    endpoint_rows = []
    for objective in (*RAW_OBJECTIVES, *CONTROL_OBJECTIVES):
        source = raw_rows if objective in RAW_OBJECTIVES else control_rows
        for condition in ("clean", "noise1x"):
            selected = _select(source, objective, condition, 7)
            endpoint_rows.append({
                "objective": objective,
                "label": LABELS[objective],
                "condition": condition,
                **{
                    key: _median(selected, key)
                    for key, _ in METRICS
                },
            })

    best_rows = []
    for objective in RAW_OBJECTIVES:
        for condition in ("clean", "noise1x"):
            for key, label in METRICS:
                medians = {
                    basis_count: _median(
                        _select(raw_rows, objective, condition, basis_count),
                        key,
                    )
                    for basis_count in range(1, 8)
                }
                best_basis = min(medians, key=medians.get)
                best_rows.append({
                    "objective": objective,
                    "label": LABELS[objective],
                    "condition": condition,
                    "metric": key,
                    "metric_label": label,
                    "best_basis_count": best_basis,
                    "best_median_mpa": medians[best_basis],
                    "bf7_median_mpa": medians[7],
                    "bf7_minus_best_mpa": medians[7] - medians[best_basis],
                })

    transition_rows = []
    for objective in RAW_OBJECTIVES:
        for condition in ("clean", "noise1x"):
            for key, label in METRICS:
                deltas = []
                for seed in (0, 1):
                    parent = _single(raw_rows, objective, condition, seed, 6)
                    child = _single(raw_rows, objective, condition, seed, 7)
                    deltas.append(float(getattr(child, key) - getattr(parent, key)))
                transition_rows.append({
                    "objective": objective,
                    "label": LABELS[objective],
                    "condition": condition,
                    "metric": key,
                    "metric_label": label,
                    "improved_count": sum(delta < 0.0 for delta in deltas),
                    "transition_count": len(deltas),
                    "median_delta_mpa": float(np.median(deltas)),
                    "seed_deltas_mpa": deltas,
                })

    return {
        "campaign": "hybrid_objective_raw_bf7_confirm_20260829",
        "case_count": 8,
        "state_count": 56,
        "endpoint_medians": endpoint_rows,
        "best_basis_by_metric": best_rows,
        "bf6_to_bf7": transition_rows,
        "provisional_targets_mpa": {
            "yielded_rmse_mpa": 20.0,
            "high_plastic_rmse_mpa": 30.0,
        },
        "decision": (
            "The raw objectives are credible clean-data training objectives, "
            "but neither is robust enough under 1x noise to choose model order "
            "from training progress. Raw 7/57 is the stronger noisy candidate; "
            "raw 7/29/57 is the stronger clean endpoint. A separate common, "
            "noise-calibrated selector is required before production use."
        ),
    }


def _executive_page(pdf: PdfPages, summary: dict[str, object]) -> None:
    figure = plt.figure(figsize=(11.69, 8.27))
    figure.suptitle(
        "Notched-EBW repaired raw BF7 confirmation",
        fontsize=19,
        y=0.96,
    )
    figure.text(
        0.06,
        0.885,
        "Decision",
        fontsize=14,
        weight="bold",
    )
    figure.text(
        0.06,
        0.825,
        textwrap.fill(str(summary["decision"]), 118),
        fontsize=11.2,
        va="top",
    )

    columns = ["Objective", "Data", "Yielded", "High plastic", "ROI"]
    cells = []
    for row in summary["endpoint_medians"]:
        cells.append([
            row["label"],
            "1× noise" if row["condition"] == "noise1x" else "Clean",
            f"{row['yielded_rmse_mpa']:.2f}",
            f"{row['high_plastic_rmse_mpa']:.2f}",
            f"{row['roi_rmse_mpa']:.2f}",
        ])
    axis = figure.add_axes((0.06, 0.34, 0.88, 0.38))
    axis.axis("off")
    table = axis.table(
        cellText=cells,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        colWidths=(0.28, 0.15, 0.15, 0.17, 0.15),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.6)
    table.scale(1.0, 1.55)
    for column in range(len(columns)):
        table[(0, column)].set_text_props(weight="bold")
        table[(0, column)].set_facecolor("0.90")

    notes = [
        "Values are BF7 median RMSE [MPa] over two matched seeds.",
        "Clean raw 7/29/57 meets the provisional 20 MPa yielded and 30 MPa high-plastic targets; raw 7/57 is marginally above 20 MPa yielded.",
        "At 1× noise, neither raw objective meets the targets. Raw 7/57 is materially better than raw 7/29/57 and both reused controls at BF7.",
        "The direct-fit BF7 reference is 3.24 MPa yielded and 1.67 MPa high-plastic, so representation capacity is not the limiting factor.",
    ]
    y = 0.285
    for note in notes:
        figure.text(0.07, y, "• " + textwrap.fill(note, 112), fontsize=10.3)
        y -= 0.055
    pdf.savefig(figure)
    plt.close(figure)


def _trajectory_page(
    pdf: PdfPages,
    rows: list[Row],
    direct: list[dict[str, str]],
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.69, 8.27),
        constrained_layout=True,
        sharex=True,
    )
    for column, condition in enumerate(("clean", "noise1x")):
        for row_index, (key, label) in enumerate(METRICS[:2]):
            axis = axes[row_index, column]
            for objective in (*CONTROL_OBJECTIVES, *RAW_OBJECTIVES):
                selected = [
                    row
                    for row in rows
                    if row.objective == objective and row.condition == condition
                ]
                medians = [
                    _median(
                        [row for row in selected if row.basis_count == basis_count],
                        key,
                    )
                    for basis_count in range(1, 8)
                ]
                axis.plot(
                    range(1, 8),
                    medians,
                    marker="o" if objective in RAW_OBJECTIVES else None,
                    linewidth=2.0 if objective in RAW_OBJECTIVES else 1.1,
                    linestyle="-" if objective in RAW_OBJECTIVES else "--",
                    color=COLORS[objective],
                    label=LABELS[objective],
                )
            axis.plot(
                [int(item["basis_count"]) for item in direct if int(item["basis_count"]) <= 7],
                [float(item[key]) for item in direct if int(item["basis_count"]) <= 7],
                color="black",
                linestyle=":",
                linewidth=1.2,
                label="Direct fit",
            )
            axis.axhline(
                20.0 if key == "yielded_rmse_mpa" else 30.0,
                color="tab:red",
                alpha=0.6,
                linewidth=1.0,
                label="Provisional target",
            )
            axis.set_title(
                f"{'Clean' if condition == 'clean' else '1× noise'} — {label}"
            )
            axis.set_ylabel("RMSE [MPa]")
            axis.grid(alpha=0.25)
            axis.set_xticks(range(1, 8))
    for axis in axes[-1]:
        axis.set_xlabel("Basis-function count")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8.5)
    figure.suptitle(
        "Median recovery trajectories: late noisy improvement is metric-dependent",
        fontsize=16,
    )
    pdf.savefig(figure)
    plt.close(figure)


def _decision_page(pdf: PdfPages, summary: dict[str, object]) -> None:
    figure = plt.figure(figsize=(11.69, 8.27))
    figure.suptitle("Model-order evidence and next steps", fontsize=18, y=0.95)

    figure.text(0.06, 0.885, "Best median BF by metric", fontsize=13, weight="bold")
    best_noise = [
        row
        for row in summary["best_basis_by_metric"]
        if row["condition"] == "noise1x"
    ]
    best_cells = [
        [
            row["label"],
            row["metric_label"],
            f"BF{row['best_basis_count']}",
            f"{row['best_median_mpa']:.2f}",
            f"{row['bf7_median_mpa']:.2f}",
            f"{row['bf7_minus_best_mpa']:+.2f}",
        ]
        for row in best_noise
    ]
    axis = figure.add_axes((0.06, 0.57, 0.88, 0.28))
    axis.axis("off")
    table = axis.table(
        cellText=best_cells,
        colLabels=("Objective", "Metric", "Best", "Best RMSE", "BF7 RMSE", "BF7−best"),
        cellLoc="center",
        loc="center",
        colWidths=(0.25, 0.20, 0.09, 0.14, 0.14, 0.13),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1.0, 1.35)
    for column in range(6):
        table[(0, column)].set_text_props(weight="bold")
        table[(0, column)].set_facecolor("0.90")

    transition = summary["bf6_to_bf7"]
    yielded_improved = sum(
        row["improved_count"]
        for row in transition
        if row["metric"] == "yielded_rmse_mpa"
    )
    high_improved = sum(
        row["improved_count"]
        for row in transition
        if row["metric"] == "high_plastic_rmse_mpa"
    )
    roi_improved = sum(
        row["improved_count"]
        for row in transition
        if row["metric"] == "roi_rmse_mpa"
    )
    figure.text(
        0.07,
        0.525,
        f"Across the eight paired BF6→BF7 transitions: yielded improved {yielded_improved}/8, "
        f"high-plastic improved {high_improved}/8, but whole-ROI improved only {roi_improved}/8.",
        fontsize=10.5,
    )

    steps = [
        ("1. Freeze a common selector before more training variants.",
         "Replay fixed raw fine-scale onset/developed evidence, broad EGI and FRE guards, and restart/map stability on every BF1–BF7 parent-child pair. Calibrate thresholds from propagated noise; use truth only for labels."),
        ("2. Challenge rejection specificity.",
         "Add the stored BF7→BF8 adverse states and controlled duplicate, unyielded, boundary/noise, and yield-hardening compensation candidates. BF7 acceptance alone is not enough."),
        ("3. Run the native-DOF audit as a non-changing BF0–BF3 smoke test.",
         "Feed frozen homogeneous regimes and calibrated residual-noise arrays into the canonical residual layout; verify restoration, ranks, group correlations and response-to-noise persistence."),
        ("4. Nominate raw 7/57 for the noisy training control.",
         "It is the stronger noisy endpoint, but retain raw 7/29/57 and equal 7/29/57 as matched controls. Do not declare a production objective from two seeds."),
        ("5. Improve proposal/search only after the selector is frozen.",
         "Add response-novelty gating to basis growth, then compare ordinary and sensitivity-SVD pattern polling on identical fixed geometries and starts."),
        ("6. Hold experimental identification.",
         "The 1×-noise synthetic results remain above the provisional 20/30 MPa targets and show late metric trade-offs. Uncertainty and stopping are not yet robust enough."),
    ]
    y = 0.46
    for heading, body in steps:
        figure.text(0.065, y, heading, fontsize=10.5, weight="bold", va="top")
        y -= 0.026
        figure.text(0.08, y, textwrap.fill(body, 128), fontsize=8.8, va="top")
        y -= 0.045
    pdf.savefig(figure)
    plt.close(figure)


def _select(
    rows: list[Row],
    objective: str,
    condition: str,
    basis_count: int,
) -> list[Row]:
    selected = [
        row
        for row in rows
        if row.objective == objective
        and row.condition == condition
        and row.basis_count == basis_count
    ]
    if not selected:
        raise ValueError(
            f"No rows for objective={objective}, condition={condition}, "
            f"BF={basis_count}."
        )
    return selected


def _single(
    rows: list[Row],
    objective: str,
    condition: str,
    seed: int,
    basis_count: int,
) -> Row:
    selected = [
        row
        for row in rows
        if row.objective == objective
        and row.condition == condition
        and row.seed == seed
        and row.basis_count == basis_count
    ]
    if len(selected) != 1:
        raise ValueError("Expected exactly one matched trajectory state.")
    return selected[0]


def _median(rows: list[Row], key: str) -> float:
    return float(np.median([getattr(row, key) for row in rows]))


def _markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Repaired raw BF7 confirmation findings",
        "",
        str(summary["decision"]),
        "",
        "## BF7 median RMSE",
        "",
        "| Objective | Data | Yielded [MPa] | High plastic [MPa] | ROI [MPa] |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["endpoint_medians"]:
        lines.append(
            f"| {row['label']} | {row['condition']} | "
            f"{row['yielded_rmse_mpa']:.2f} | "
            f"{row['high_plastic_rmse_mpa']:.2f} | "
            f"{row['roi_rmse_mpa']:.2f} |"
        )
    lines.extend([
        "",
        "## Next decision",
        "",
        "Freeze and validate a truth-free, noise-calibrated model-order selector on the complete BF1–BF7 trajectories and BF8/adverse states. In parallel, run the canonical native-DOF audit as a non-changing BF0–BF3 smoke test. Do not release the method to experimental identification yet.",
        "",
    ])
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--direct-fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
