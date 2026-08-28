"""Create a concise PDF report from a completed gate-campaign analysis."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def main() -> None:
    args = _parse_args()
    analysis = args.analysis.expanduser().resolve()
    campaign = analysis.parent
    states = _read_csv(analysis / "state_metrics.csv")
    gates = _read_csv(analysis / "gate_summary.csv")
    ranks = json.loads((analysis / "rank_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((campaign / "campaign_manifest.json").read_text(encoding="utf-8"))
    output = args.output or analysis / "NOTCHED_EBW_GATE_CAMPAIGN.pdf"
    _write_pdf(output, manifest, states, gates, ranks, analysis)
    print(output)


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            {key: _parse(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def _parse(value: str) -> object:
    if value in {"True", "False"}:
        return value == "True"
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def _write_pdf(output, manifest, states, gates, ranks, analysis) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        _title_page(pdf, manifest, states, gates, ranks)
        _gate_page(pdf, gates)
        _model_order_page(pdf, states)
        _paired_increment_page(pdf, states)
        _append_png(pdf, analysis / "objective_vs_yielded_rmse.png")
        _append_png(pdf, analysis / "gate_comparison.png")


def _title_page(pdf, manifest, states, gates, ranks) -> None:
    figure = plt.figure(figsize=(11.69, 8.27))
    figure.text(0.06, 0.91, "Notched-EBW gate campaign", fontsize=22, weight="bold")
    completed = sum(bool(row["is_final_accepted"]) for row in states)
    campaign_states = sum(row["source"] == "campaign" for row in states)
    rows = [row for row in gates if row["policy"] == "sensitivity_correction"]
    lines = [
        f"Campaign: {manifest['campaign_name']}",
        f"Generated: {datetime.now().astimezone():%Y-%m-%d %H:%M %Z}",
        f"Completed final accepted runs: {completed}; analysed solve states: {campaign_states}",
        "",
        "Interpretation rules:",
        "• Mechanical objective is a closure measure; judge map recovery separately.",
        "• Compare model order with paired changes within each optimiser seed.",
        "• Do not select a production gate from one deterministic endpoint.",
    ]
    if rows:
        lead = min(rows, key=lambda row: row["median_yielded_rmse_mpa"])
        lines.extend([
            "",
            f"Current synthetic lead: {100 * lead['gate']:.1f}% gate, median yielded RMSE "
            f"{lead['median_yielded_rmse_mpa']:.2f} MPa.",
        ])
    mechanical = ranks["campaign_all_solves"]["objective"]["yielded_rmse_mpa"]["spearman_r"]
    active = ranks["campaign_all_solves"]["active_objective"]["yielded_rmse_mpa"]["spearman_r"]
    lines.append(
        f"Spearman objective/yielded-RMSE correlation: mechanical {mechanical:.3f}; "
        f"sensitivity-active {active:.3f}."
    )
    figure.text(0.07, 0.76, "\n".join(lines), fontsize=12, va="top", linespacing=1.65)
    pdf.savefig(figure)
    plt.close(figure)


def _gate_page(pdf, gates) -> None:
    figure, axis = plt.subplots(figsize=(11.69, 8.27))
    axis.axis("off")
    axis.set_title("Final accepted gate summary", fontsize=19, pad=24)
    columns = ["Policy", "Gate", "Runs", "Median J", "Yielded RMSE", "IQR", ">10% points", "Median BFs"]
    table_rows = [
        [
            str(row["policy"]), f"{100 * row['gate']:.1f}%", f"{row['runs']:.0f}",
            f"{row['median_objective']:.5f}", f"{row['median_yielded_rmse_mpa']:.2f}",
            f"{row['iqr_yielded_rmse_mpa']:.2f}", f"{100 * row['median_yielded_above_10pct']:.1f}%",
            f"{row['median_basis_count']:.1f}",
        ]
        for row in gates
    ]
    table = axis.table(cellText=table_rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)
    axis.text(
        0.05, 0.14,
        "Use this table only for endpoint comparison. The next pages test whether each extra basis\n"
        "repeatedly improves active-region recovery, rather than merely reducing the scalar objective.",
        transform=axis.transAxes, fontsize=11,
    )
    pdf.savefig(figure)
    plt.close(figure)


def _model_order_page(pdf, states) -> None:
    rows = [
        row for row in states
        if row["source"] == "campaign" and row["accepted"] is True
    ]
    groups = _groups(rows, ("policy", "gate", "basis_count"))
    figure, axes = plt.subplots(1, 2, figsize=(11.69, 5.2), layout="constrained")
    for (policy, gate), colour in zip(
        sorted({(row["policy"], row["gate"]) for row in rows}),
        plt.rcParams["axes.prop_cycle"].by_key()["color"],
        strict=False,
    ):
        series = [
            (basis, values)
            for (item_policy, item_gate, basis), values in groups.items()
            if item_policy == policy and item_gate == gate
        ]
        series.sort(key=lambda item: item[0])
        basis = np.asarray([item[0] for item in series])
        objective = np.asarray([np.median([row["objective"] for row in item[1]]) for item in series])
        yielded = np.asarray([np.median([row["yielded_rmse_mpa"] for row in item[1]]) for item in series])
        label = f"{policy}, {100 * gate:.1f}%"
        axes[0].plot(basis, objective, "o-", label=label, color=colour)
        axes[1].plot(basis, yielded, "o-", label=label, color=colour)
    axes[0].set(xlabel="Accepted basis count", ylabel="Median mechanical objective")
    axes[1].set(xlabel="Accepted basis count", ylabel="Median yielded RMSE [MPa]")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    figure.suptitle("Accepted model-order trajectory", fontsize=17)
    pdf.savefig(figure)
    plt.close(figure)


def _paired_increment_page(pdf, states) -> None:
    rows = [
        row for row in states
        if row["source"] == "campaign" and row["accepted"] is True
    ]
    figure, axis = plt.subplots(figsize=(11.69, 8.27))
    axis.axis("off")
    axis.set_title("Paired effect of adding a basis", fontsize=19, pad=24)
    table_rows = []
    for policy, gate in sorted({(row["policy"], row["gate"]) for row in rows}):
        by_seed = _groups(
            [row for row in rows if row["policy"] == policy and row["gate"] == gate],
            ("seed", "basis_count"),
        )
        basis_counts = sorted({key[1] for key in by_seed})
        for lower, upper in zip(basis_counts, basis_counts[1:], strict=False):
            changes = []
            for seed in {key[0] for key in by_seed}:
                before = by_seed.get((seed, lower), [])
                after = by_seed.get((seed, upper), [])
                if before and after:
                    changes.append(after[-1]["yielded_rmse_mpa"] - before[-1]["yielded_rmse_mpa"])
            if changes:
                table_rows.append([
                    policy, f"{100 * gate:.1f}%", f"{lower:.0f}→{upper:.0f}",
                    f"{len(changes)}", f"{np.median(changes):+.2f}",
                    f"{sum(change < 0.0 for change in changes)}/{len(changes)}",
                ])
    columns = ["Policy", "Gate", "Bases", "Pairs", "Median Δ yielded RMSE [MPa]", "Seeds improved"]
    table = axis.table(cellText=table_rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.7)
    axis.text(
        0.06, 0.13,
        "Negative Δ means the additional basis improves yielded-region recovery. A basis should not be retained\n"
        "as a production default if it mainly reduces mechanical objective while this paired metric is inconsistent or positive.",
        transform=axis.transAxes, fontsize=11,
    )
    pdf.savefig(figure)
    plt.close(figure)


def _groups(rows, keys):
    output: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        output.setdefault(tuple(row[key] for key in keys), []).append(row)
    return output


def _append_png(pdf, path: Path) -> None:
    if not path.is_file():
        return
    image = plt.imread(path)
    figure, axis = plt.subplots(figsize=(11.69, 8.27))
    axis.imshow(image)
    axis.axis("off")
    pdf.savefig(figure)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
