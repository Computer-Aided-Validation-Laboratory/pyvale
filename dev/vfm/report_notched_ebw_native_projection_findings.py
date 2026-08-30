"""Report findings from the partial Notched-EBW native-projection noise study."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr


INK = "#17324d"
MUTED = "#536878"
BLUE = "#3478a6"
ORANGE = "#d9822b"
GREEN = "#26866a"
RED = "#b54a4a"
COMPONENTS = ("raw_rms", "projected_rms", "yield_unique_rms", "hardening_unique_rms")
TARGETS = ("yielded_rmse_mpa", "high_plastic_rmse_mpa")
TARGET_LABEL = {
    "yielded_rmse_mpa": "Yielded-region map error",
    "high_plastic_rmse_mpa": "High-plastic-region map error",
}


def _read_scores(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("state") == "state":
                continue
            for key in ("seed", "basis_count", "noise_replicate", "native_dofs"):
                row[key] = int(row[key])
            for key in ("noise_scale", *TARGETS, *COMPONENTS):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def _read_summary(path: Path) -> tuple[list[dict[str, object]], int]:
    rows, seen, physical_rows = [], set(), 0
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("noise_scale") == "noise_scale":
                continue
            physical_rows += 1
            key = tuple(row[name] for name in
                        ("noise_scale", "noise_replicate", "block", "target", "component"))
            if key in seen:
                continue
            seen.add(key)
            row["noise_scale"] = float(row["noise_scale"])
            row["noise_replicate"] = int(row["noise_replicate"])
            row["spearman_r"] = float(row["spearman_r"])
            row["pairwise_accuracy"] = float(row["pairwise_accuracy"])
            rows.append(row)
    return rows, physical_rows


def _aggregate(summary: list[dict[str, object]]) -> dict[tuple, dict[str, float]]:
    grouped: dict[tuple, list[list[float]]] = defaultdict(lambda: [[], []])
    for row in summary:
        key = (row["target"], row["noise_scale"], row["block"], row["component"])
        if np.isfinite(row["spearman_r"]):
            grouped[key][0].append(row["spearman_r"])
        if np.isfinite(row["pairwise_accuracy"]):
            grouped[key][1].append(row["pairwise_accuracy"])
    return {
        key: {
            "rho": float(np.mean(values[0])) if values[0] else np.nan,
            "rho_std": float(np.std(values[0])) if values[0] else np.nan,
            "pairwise": float(np.mean(values[1])) if values[1] else np.nan,
            "n": len(values[0]),
        }
        for key, values in grouped.items()
    }


def _new_page(title: str, subtitle: str = "") -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    ax = fig.add_axes((0.065, 0.065, 0.87, 0.82)); ax.axis("off")
    fig.text(0.065, 0.945, title, fontsize=21, weight="bold", color=INK)
    if subtitle:
        fig.text(0.065, 0.908, subtitle, fontsize=10.5, color=MUTED)
    return fig, ax


def _save(pdf: PdfPages, fig: plt.Figure) -> None:
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _title_page(pdf: PdfPages, scores, summary, physical_rows) -> None:
    states = {(r["state"], r["seed"], r["basis_count"], r["yielded_rmse_mpa"],
               r["high_plastic_rmse_mpa"]) for r in scores if r["noise_scale"] == 0}
    fig, ax = _new_page("Native-projection residual study: BF5–6 findings")
    ax.text(0, .88, "WDBN1-calibrated synthetic strain noise · transferred workstation subset",
            fontsize=14, weight="bold", transform=ax.transAxes)
    ax.text(0, .75,
            f"Coverage: {len(states)} states = BF5 and BF6 for all eight optimiser seeds.\n"
            f"Evidence: {len(scores):,} state/block/noise rows; {len(summary):,} unique ranking summaries; "
            "128 replicates at each non-zero noise scale.",
            fontsize=12, linespacing=1.55, transform=ax.transAxes)
    boxes = [
        (0.02, .47, "YIELDED-REGION", "EGI7 late plasticity\nprojected RMS", "ρ = 0.805 at 1× noise", GREEN),
        (.35, .47, "HIGH-PLASTIC", "EGI7 yield onset\nraw RMS", "ρ = 0.939 at 1× noise", BLUE),
        (.68, .47, "NOT SUPPORTED", "FRE components", "best ρ = 0.332 / −0.017", RED),
    ]
    for x, y, heading, name, value, colour in boxes:
        ax.text(x, y, f"{heading}\n{name}\n{value}", transform=ax.transAxes,
                fontsize=12, weight="bold", color=colour, linespacing=1.45,
                bbox=dict(boxstyle="round,pad=.65", fc="#f5f8fa", ec="#c4d0d8"))
    ax.text(0, .20,
            "Main interpretation: the native projection is useful, but its strongest late-plasticity "
            "projection and yield-unique reductions carry almost the same ranking information. The "
            "complementary piece is raw, fine-scale EGI at yield onset, which is especially sensitive "
            "to high-plastic-region map quality.",
            fontsize=12.2, linespacing=1.5, transform=ax.transAxes, wrap=True)
    if physical_rows > len(summary):
        ax.text(0, .07,
                f"Data note: the transferred summary CSV contains a duplicated block ({physical_rows:,} data lines); "
                f"this report deduplicates it to {len(summary):,} unique keys before analysis.",
                fontsize=9.5, color=MUTED, transform=ax.transAxes)
    ax.text(0, .01, f"Generated {datetime.now().astimezone():%Y-%m-%d %H:%M %Z}",
            fontsize=9, color=MUTED, transform=ax.transAxes)
    _save(pdf, fig)


def _coverage_page(pdf: PdfPages, scores) -> None:
    states = {}
    for row in scores:
        if row["noise_scale"] == 0:
            states[row["state"]] = row
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    fig.suptitle("State-library coverage and BF5 → BF6 improvement", x=.065, ha="left",
                 fontsize=20, weight="bold", color=INK)
    axes = fig.subplots(1, 2, gridspec_kw={"left": .08, "right": .96, "bottom": .18,
                                           "top": .83, "wspace": .25})
    for ax, target, title in zip(axes, TARGETS,
                                 ("Yielded-region RMSE [MPa]", "High-plastic-region RMSE [MPa]"), strict=True):
        for seed in range(8):
            points = sorted((r["basis_count"], r[target]) for r in states.values() if r["seed"] == seed)
            ax.plot(*zip(*points, strict=True), marker="o", alpha=.8, label=f"seed {seed}")
        ax.set(xticks=[5, 6], xlabel="Basis functions", ylabel=title)
        ax.grid(alpha=.25); ax.set_axisbelow(True)
    axes[1].legend(ncol=2, fontsize=8, frameon=False)
    for target, y in zip(TARGETS, (.105, .065), strict=True):
        bf5 = [r[target] for r in states.values() if r["basis_count"] == 5]
        bf6 = [r[target] for r in states.values() if r["basis_count"] == 6]
        fig.text(.08, y,
                 f"{TARGET_LABEL[target]}: median {np.median(bf5):.2f} → {np.median(bf6):.2f} MPa; "
                 f"BF6 improves all 8 paired seeds.", fontsize=10.5, color=INK)
    _save(pdf, fig)


def _ranking_page(pdf: PdfPages, aggregate) -> None:
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    fig.suptitle("Best component rankings at the calibrated 1× noise level", x=.055,
                 ha="left", fontsize=19, weight="bold", color=INK)
    axes = fig.subplots(1, 2, gridspec_kw={"left": .20, "right": .98, "bottom": .10,
                                           "top": .84, "wspace": .72})
    for ax, target in zip(axes, TARGETS, strict=True):
        values = [(v["rho"], v["pairwise"], k[2], k[3]) for k, v in aggregate.items()
                  if k[0] == target and k[1] == 1.0 and np.isfinite(v["rho"])]
        values.sort(reverse=True); values = values[:12][::-1]
        labels = [f"{block.replace('__', ' · ')}\n{component.replace('_rms', '')}" for _, _, block, component in values]
        bars = ax.barh(np.arange(len(values)), [v[0] for v in values], color=BLUE)
        ax.set_yticks(np.arange(len(values)), labels, fontsize=7.7)
        ax.set(xlim=(0, 1), xlabel="Mean Spearman ρ", title=TARGET_LABEL[target])
        ax.grid(axis="x", alpha=.25); ax.set_axisbelow(True)
        for bar, value in zip(bars, values, strict=True):
            ax.text(value[0] + .012, bar.get_y()+bar.get_height()/2,
                    f"{value[0]:.3f} / {value[1]:.3f}", va="center", fontsize=7.5)
    fig.text(.55, .035, "Labels at bar ends: mean Spearman ρ / mean pairwise ranking accuracy",
             ha="center", fontsize=9.5, color=MUTED)
    _save(pdf, fig)


def _heatmap_page(pdf: PdfPages, aggregate) -> None:
    blocks = [f"egi{w}__{regime}" for w in (7, 15, 29, 57)
              for regime in ("pre_yield", "yield_onset", "developed_plasticity", "late_plasticity")]
    blocks += [f"fre__{regime}" for regime in
               ("pre_yield", "yield_onset", "developed_plasticity", "late_plasticity")]
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), constrained_layout=True)
    fig.suptitle("Where the ranking information resides at 1× noise", fontsize=19,
                 weight="bold", color=INK)
    image = None
    for ax, target in zip(axes, TARGETS, strict=True):
        matrix = np.array([[aggregate.get((target, 1.0, block, comp), {}).get("rho", np.nan)
                            for comp in COMPONENTS] for block in blocks])
        image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(4), ["raw", "projected", "yield unique", "hardening unique"],
                      rotation=28, ha="right", fontsize=8)
        ax.set_yticks(range(len(blocks)), [b.replace("__", " · ") for b in blocks], fontsize=7)
        ax.set_title(TARGET_LABEL[target], fontsize=12)
        for i in range(len(blocks)):
            for j in range(4):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=5.8,
                            color="white" if abs(matrix[i,j]) > .62 else "black")
    fig.colorbar(image, ax=axes, label="Mean Spearman ρ", shrink=.8)
    _save(pdf, fig)


def _noise_page(pdf: PdfPages, aggregate) -> None:
    selected = [
        ("EGI7 onset raw", "egi7__yield_onset", "raw_rms", BLUE),
        ("EGI7 late projected", "egi7__late_plasticity", "projected_rms", GREEN),
        ("EGI7 late yield unique", "egi7__late_plasticity", "yield_unique_rms", ORANGE),
        ("EGI15 onset raw", "egi15__yield_onset", "raw_rms", "#8064a2"),
        ("FRE onset raw", "fre__yield_onset", "raw_rms", RED),
    ]
    scales = (0.0, .5, 1.0, 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), constrained_layout=True)
    fig.suptitle("Ranking robustness to WDBN1-calibrated noise", fontsize=20,
                 weight="bold", color=INK)
    for ax, target in zip(axes, TARGETS, strict=True):
        for label, block, component, colour in selected:
            y = [aggregate.get((target, scale, block, component), {}).get("rho", np.nan)
                 for scale in scales]
            ax.plot(scales, y, marker="o", label=label, color=colour)
        ax.axhline(0, color="black", lw=.7)
        ax.set(xlabel="Noise scale", ylabel="Mean Spearman ρ", ylim=(-.3, 1),
               title=TARGET_LABEL[target], xticks=scales)
        ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8)
    _save(pdf, fig)


def _complementarity_page(pdf: PdfPages, scores) -> None:
    selected = {
        "Onset raw": ("egi7__yield_onset", "raw_rms"),
        "Late projected": ("egi7__late_plasticity", "projected_rms"),
        "Late yield unique": ("egi7__late_plasticity", "yield_unique_rms"),
        "Developed yield unique": ("egi7__developed_plasticity", "yield_unique_rms"),
        "FRE onset raw": ("fre__yield_onset", "raw_rms"),
    }
    indexed = defaultdict(dict)
    for row in scores:
        if row["noise_scale"] != 1.0:
            continue
        for name, (block, component) in selected.items():
            if row["block"] == block:
                indexed[(row["noise_replicate"], row["state"])][name] = row[component]
    names = list(selected); correlations = np.zeros((len(names), len(names)))
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            values = []
            for replicate in range(128):
                keys = sorted(key for key in indexed if key[0] == replicate)
                x = [indexed[key][left] for key in keys]; y = [indexed[key][right] for key in keys]
                values.append(spearmanr(x, y).statistic)
            correlations[i, j] = np.nanmean(values)
    fig, ax = plt.subplots(figsize=(11.69, 8.27), constrained_layout=True)
    fig.suptitle("Complementarity of leading scalar reductions", fontsize=20,
                 weight="bold", color=INK)
    image = ax.imshow(correlations, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=25, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_title(
        "Late projection reductions are redundant; FRE is independent but weak",
        fontsize=11, pad=13, color=MUTED,
    )
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{correlations[i,j]:.3f}", ha="center", va="center",
                    color="white" if abs(correlations[i,j]) > .65 else "black")
    fig.colorbar(image, ax=ax, label="Mean cross-state Spearman ρ at 1× noise", shrink=.78)
    _save(pdf, fig)


def _conclusion_page(pdf: PdfPages) -> None:
    fig, ax = _new_page("Conclusions and next step")
    sections = [
        ("Carry forward", [
            "EGI7 yield-onset raw RMS as the strongest high-plastic-region sentinel.",
            "One late/developed EGI7 projection term for yielded-region discrimination; projected RMS and yield-unique RMS should not both receive full weight.",
            "EGI15 as a secondary scale only until BF7–8 validation shows independent value.",
        ]),
        ("Do not carry forward yet", [
            "FRE as a map-discrimination term in its present scalar form; it remains useful for force consistency and physical guarding.",
            "The pre-yield hardening-unique ranking despite high pairwise accuracy: its physical interpretation is doubtful and it may reflect BF5/BF6 or hardening-compensation structure.",
            "A production objective change based on this partial BF5–6 library.",
        ]),
        ("Smallest decisive follow-up", [
            "Resume only the missing BF7–8 native-projection states on the workstation, preserving the current checkpoints.",
            "Recompute rankings within BF5–8 and, critically, within BF7–8 alone so gross BF-stage separation cannot dominate.",
            "Test a compact selector with EGI7 onset raw + one EGI7 late projection + EGI15 guard, then validate across matched seeds and calibrated noise.",
        ]),
    ]
    y = .89
    for heading, bullets in sections:
        ax.text(0, y, heading, fontsize=14, weight="bold", color=INK, transform=ax.transAxes); y -= .05
        for bullet in bullets:
            ax.text(.025, y, f"• {bullet}", fontsize=10.5, transform=ax.transAxes,
                    va="top", wrap=True); y -= .065
        y -= .015
    ax.text(0, .015,
            "Decision: the evidence advances objective design, but BF7–8 completion is required before tuning weights.",
            fontsize=12.2, weight="bold", color=GREEN, transform=ax.transAxes)
    _save(pdf, fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    output = args.output or root / "NOTCHED_EBW_NATIVE_PROJECTION_NOISE_FINDINGS.pdf"
    scores = _read_scores(root / "projection_noise_scores.csv")
    summary, physical_rows = _read_summary(root / "projection_noise_summary.csv")
    aggregate = _aggregate(summary)
    with PdfPages(output) as pdf:
        _title_page(pdf, scores, summary, physical_rows)
        _coverage_page(pdf, scores)
        _ranking_page(pdf, aggregate)
        _heatmap_page(pdf, aggregate)
        _noise_page(pdf, aggregate)
        _complementarity_page(pdf, scores)
        _conclusion_page(pdf)
        info = pdf.infodict()
        info["Title"] = "Notched-EBW native-projection/noise findings"
        info["Author"] = "PyVale VFM investigation"
        info["Subject"] = "BF5–6 residual projection and WDBN1-calibrated noise study"
    print(output)


if __name__ == "__main__":
    main()
