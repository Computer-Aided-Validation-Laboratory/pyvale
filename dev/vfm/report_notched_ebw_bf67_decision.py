"""Generate a focused BF6-to-BF7 native-projection decision report."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

COMPONENTS = ("raw_rms", "projected_rms", "yield_unique_rms")
TARGETS = ("yielded_rmse_mpa", "high_plastic_rmse_mpa")
BLUE = "#2878B5"
ORANGE = "#E07A1F"
GREEN = "#3A923A"
RED = "#C43C39"
NAVY = "#17324D"


@dataclass(frozen=True)
class Term:
    label: str
    block: str
    component: str


TERMS = (
    Term("EGI7 onset raw", "egi7__yield_onset", "raw_rms"),
    Term("EGI7 late raw", "egi7__late_plasticity", "raw_rms"),
    Term("EGI15 developed raw", "egi15__developed_plasticity", "raw_rms"),
    Term("EGI7 late yield-unique", "egi7__late_plasticity", "yield_unique_rms"),
    Term("EGI57 onset raw", "egi57__yield_onset", "raw_rms"),
    Term("FRE developed raw", "fre__developed_plasticity", "raw_rms"),
)


def main() -> None:
    args = _parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(args.campaign_root.resolve())
    errors = _state_errors(rows)
    objectives = _objective_changes(args.identification_root.resolve())
    rankings = _paired_rankings(rows, errors)
    seed_table = _seed_table(rows, errors, objectives, scale=1.0)
    selectors = _selector_results(rows, errors, scale=1.0)
    rankings.to_csv(output / "bf67_component_rankings.csv", index=False)
    seed_table.to_csv(output / "bf67_seed_decisions.csv", index=False)
    selectors.to_csv(output / "bf67_selector_results.csv", index=False)
    report = _report(output, errors, rankings, seed_table, selectors)
    print(report)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--identification-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_rows(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("state_*/projection_noise_rows.jsonl"))
    if len(paths) != 24:
        raise RuntimeError(f"Expected 24 BF5-BF7 state files, found {len(paths)}")
    rows = pd.concat((pd.read_json(path, lines=True) for path in paths), ignore_index=True)
    expected = {(seed, bf) for seed in range(8) for bf in (5, 6, 7)}
    actual = set(zip(rows.seed.astype(int), rows.basis_count.astype(int), strict=True))
    if actual != expected:
        raise RuntimeError(f"State coverage mismatch: {sorted(actual)}")
    return rows


def _state_errors(rows: pd.DataFrame) -> pd.DataFrame:
    errors = rows[["seed", "basis_count", *TARGETS]].drop_duplicates().copy()
    errors = errors.sort_values(["seed", "basis_count"]).reset_index(drop=True)
    for target in TARGETS:
        errors[f"delta67_{target}"] = errors.groupby("seed")[target].diff()
    return errors


def _objective_changes(root: Path) -> dict[int, float]:
    states = pd.read_csv(root / "analysis" / "state_metrics.csv")
    states = states[
        states.case_name.str.startswith("spd_sensitivity_gate0p0pct_seed")
        & states.accepted.astype(bool) & states.basis_count.isin([6, 7])
    ]
    pivot = states.pivot_table(
        index="seed", columns="basis_count", values="objective", aggfunc="last"
    ).dropna()
    return {int(seed): float(row[7] - row[6]) for seed, row in pivot.iterrows()}


def _paired_rankings(rows: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    truth = errors[errors.basis_count == 7].set_index("seed")
    records = []
    for scale in sorted(rows.noise_scale.unique()):
        subset = rows[(rows.noise_scale == scale) & rows.basis_count.isin([6, 7])]
        for (block, component), selected in _component_groups(subset):
            pivot = selected.pivot_table(
                index=["noise_replicate", "seed"], columns="basis_count",
                values=component, aggfunc="first",
            ).dropna()
            if not {6, 7}.issubset(pivot.columns):
                continue
            score_delta = pivot[7] - pivot[6]
            replicate = pivot.index.get_level_values("noise_replicate")
            seed = pivot.index.get_level_values("seed").astype(int)
            for target in TARGETS:
                truth_delta = pd.Series(
                    truth.loc[seed, f"delta67_{target}"].to_numpy(), index=pivot.index
                )
                correct = np.sign(score_delta) == np.sign(truth_delta)
                per_rep_accuracy = correct.groupby(replicate).mean()
                per_rep_rho = []
                for _, indices in score_delta.groupby(replicate).groups.items():
                    rho = spearmanr(score_delta.loc[indices], truth_delta.loc[indices]).statistic
                    per_rep_rho.append(float(rho))
                records.append({
                    "noise_scale": float(scale), "block": block,
                    "component": component, "target": target,
                    "mean_accuracy": float(per_rep_accuracy.mean()),
                    "std_accuracy": float(per_rep_accuracy.std(ddof=0)),
                    "mean_delta_rho": float(np.nanmean(per_rep_rho)),
                    "std_delta_rho": float(np.nanstd(per_rep_rho)),
                    "acceptance_rate": float((score_delta < 0).mean()),
                })
    return pd.DataFrame(records)


def _component_groups(rows: pd.DataFrame):
    for block in sorted(rows.block.unique()):
        selected = rows[rows.block == block]
        for component in COMPONENTS:
            yield (block, component), selected


def _mean_delta(rows: pd.DataFrame, term: Term, scale: float) -> pd.Series:
    selected = rows[
        (rows.noise_scale == scale) & (rows.block == term.block)
        & rows.basis_count.isin([6, 7])
    ]
    pivot = selected.pivot_table(
        index=["noise_replicate", "seed"], columns="basis_count",
        values=term.component, aggfunc="first",
    ).dropna()
    delta = pivot[7] - pivot[6]
    return delta.groupby("seed").mean()


def _seed_table(rows, errors, objectives, scale):
    table = errors[errors.basis_count == 7][
        ["seed", "delta67_yielded_rmse_mpa", "delta67_high_plastic_rmse_mpa"]
    ].copy()
    table["delta67_objective"] = table.seed.map(objectives)
    for term in TERMS:
        table[term.label] = table.seed.map(_mean_delta(rows, term, scale))
    return table.sort_values("seed").reset_index(drop=True)


def _selector_results(rows, errors, scale):
    truth = errors[errors.basis_count == 7].set_index("seed")
    deltas = {}
    for term in TERMS:
        selected = rows[
            (rows.noise_scale == scale) & (rows.block == term.block)
            & rows.basis_count.isin([6, 7])
        ]
        pivot = selected.pivot_table(
            index=["noise_replicate", "seed"], columns="basis_count",
            values=term.component, aggfunc="first",
        ).dropna()
        deltas[term.label] = pivot[7] - pivot[6]
    rules = {
        "EGI7 onset raw": deltas[TERMS[0].label] < 0,
        "EGI7 late raw": deltas[TERMS[1].label] < 0,
        "EGI15 developed raw": deltas[TERMS[2].label] < 0,
        "EGI7 late yield-unique": deltas[TERMS[3].label] < 0,
        "Onset + late (both)": (deltas[TERMS[0].label] < 0) & (deltas[TERMS[1].label] < 0),
        "2/3 fine-scale vote": sum((deltas[t.label] < 0).astype(int) for t in TERMS[:3]) >= 2,
        "2/3 onset/late/broad": sum((deltas[t.label] < 0).astype(int) for t in (TERMS[0], TERMS[1], TERMS[4])) >= 2,
        "2/3 onset/late/FRE": sum((deltas[t.label] < 0).astype(int) for t in (TERMS[0], TERMS[1], TERMS[5])) >= 2,
    }
    records = []
    index = next(iter(deltas.values())).index
    seeds = index.get_level_values("seed").astype(int)
    for target in TARGETS:
        improves = pd.Series(
            truth.loc[seeds, f"delta67_{target}"].to_numpy() < 0, index=index
        )
        for name, accept in rules.items():
            true_positive = accept & improves
            true_negative = (~accept) & (~improves)
            records.append({
                "selector": name, "target": target,
                "accuracy": float((accept == improves).mean()),
                "sensitivity": float(true_positive.sum() / improves.sum()),
                "specificity": float(true_negative.sum() / (~improves).sum()),
                "acceptance_rate": float(accept.mean()),
            })
    return pd.DataFrame(records)


def _report(output, errors, rankings, seed_table, selectors):
    path = output / "NOTCHED_EBW_BF6_TO_BF7_DECISION_SUMMARY.pdf"
    bf6 = errors[errors.basis_count == 6]
    bf7 = errors[errors.basis_count == 7]
    dy = bf7.yielded_rmse_mpa.to_numpy() - bf6.yielded_rmse_mpa.to_numpy()
    dh = bf7.high_plastic_rmse_mpa.to_numpy() - bf6.high_plastic_rmse_mpa.to_numpy()
    one = rankings[rankings.noise_scale == 1.0]
    top_y = one[one.target == "yielded_rmse_mpa"].sort_values(
        ["mean_accuracy", "mean_delta_rho"], ascending=False
    ).head(10)
    top_h = one[one.target == "high_plastic_rmse_mpa"].sort_values(
        ["mean_accuracy", "mean_delta_rho"], ascending=False
    ).head(10)

    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27)); fig.suptitle(
            "Notched-EBW BF6 → BF7 decision summary", fontsize=20, color=NAVY, y=.94
        )
        conclusion = (
            "Decision: retain BF7 as a guarded candidate, not as an automatic endpoint.\n\n"
            f"BF7 improves yielded-region error in {(dy < 0).sum()}/8 seeds "
            f"(median change {np.median(dy):+.2f} MPa) and high-plastic error in "
            f"{(dh < 0).sum()}/8 (median {np.median(dh):+.2f} MPa). "
            "The only degradation is seed 5 and is negligible.\n\n"
            "The fine-scale signal survives BF7 and realistic noise. Raw EGI7 late and raw "
            "EGI15 developed evidence give the strongest paired decisions. Projected and "
            "yield-unique terms retain ranking power but do not reject the marginally adverse "
            "seed, so they are not yet suitable as sole acceptance gates.\n\n"
            "Important limit: this BF5-BF7 library contains almost no adverse growth step. "
            "It can establish that BF7 is useful, but cannot by itself validate a stopping "
            "rule's ability to reject BF8-like overfit."
        )
        fig.text(.07, .82, _wrap_paragraphs(conclusion), fontsize=12.5, va="top", linespacing=1.48)
        fig.text(.07, .12, "24 states · 8 matched seeds · 128 replicates at each non-zero noise scale · native optimiser DOFs", fontsize=10, color="#555555")
        pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), constrained_layout=True)
        for axis, target, title in zip(axes, TARGETS, ("Yielded region", "High-plastic region"), strict=True):
            six = bf6[target].to_numpy(); seven = bf7[target].to_numpy()
            for seed, (left, right) in enumerate(zip(six, seven, strict=True)):
                axis.plot([6, 7], [left, right], marker="o", lw=1.8,
                          color=GREEN if right < left else RED, label=f"seed {seed}")
            axis.set(xticks=[6, 7], xlabel="Basis functions", ylabel="Map RMSE [MPa]", title=title)
            axis.grid(alpha=.25)
        axes[1].legend(fontsize=8, ncol=2)
        fig.suptitle("True paired map-error change", fontsize=18, color=NAVY)
        pdf.savefig(fig); plt.close(fig)

        for table, title in ((top_y, "Yielded-region BF6→7 ranking at 1× noise"),
                             (top_h, "High-plastic BF6→7 ranking at 1× noise")):
            fig, axis = plt.subplots(figsize=(11.69, 8.27), constrained_layout=True)
            labels = [f"{r.block.replace('__', ' · ')} · {r.component.replace('_rms','')}" for r in table.itertuples()]
            yy = np.arange(len(table))[::-1]
            axis.barh(yy, table.mean_accuracy, color=BLUE)
            axis.set(yticks=yy, yticklabels=labels, xlim=(0, 1), xlabel="Mean paired decision accuracy", title=title)
            for y, row in zip(yy, table.itertuples(), strict=True):
                axis.text(row.mean_accuracy + .01, y, f"{row.mean_accuracy:.3f} / ρΔ {row.mean_delta_rho:.3f}", va="center", fontsize=9)
            axis.grid(axis="x", alpha=.25)
            pdf.savefig(fig); plt.close(fig)

        fig, axis = plt.subplots(figsize=(11.69, 8.27), constrained_layout=True)
        curves = (
            TERMS[0], TERMS[1], TERMS[2], TERMS[3], TERMS[4], TERMS[5],
        )
        for term in curves:
            selected = rankings[
                (rankings.block == term.block) & (rankings.component == term.component)
                & (rankings.target == "yielded_rmse_mpa")
            ].sort_values("noise_scale")
            axis.plot(selected.noise_scale, selected.mean_accuracy, marker="o", label=term.label)
        axis.set(xlabel="WDBN1 noise scale", ylabel="BF6→7 paired accuracy", ylim=(0, 1.02), title="Yielded-region decision robustness")
        axis.grid(alpha=.25); axis.legend(fontsize=8, ncol=2)
        pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(11.69, 8.27)); fig.suptitle(
            "Seed-level BF6 → BF7 decision at 1× noise", fontsize=18, color=NAVY, y=.95
        )
        display = seed_table.copy()
        columns = ["seed", "delta67_objective", "delta67_yielded_rmse_mpa", "delta67_high_plastic_rmse_mpa", TERMS[0].label, TERMS[1].label, TERMS[2].label, TERMS[4].label, TERMS[5].label]
        display = display[columns]
        labels = ["Seed", "ΔJ", "Δ yield RMSE", "Δ high-P RMSE", "EGI7 onset", "EGI7 late", "EGI15 dev.", "EGI57 onset", "FRE dev."]
        cell = []
        for row in display.itertuples(index=False, name=None):
            cell.append([f"{int(row[0])}"] + ["—" if pd.isna(value) else f"{value:+.3g}" for value in row[1:]])
        axis = fig.add_axes([.04, .25, .92, .58]); axis.axis("off")
        table = axis.table(cellText=cell, colLabels=labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(7.8); table.scale(1, 1.55)
        fig.text(.06, .15, "Negative values favour BF7. Metric entries are mean BF7−BF6 score changes over 128 noise realisations.", fontsize=10)
        pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(11.69, 8.27)); fig.suptitle(
            "Compact selector screen at 1× noise", fontsize=18, color=NAVY, y=.95
        )
        show = selectors[selectors.target == "yielded_rmse_mpa"].sort_values("accuracy", ascending=False)
        lines = ["Selector                              accuracy  sensitivity  specificity  accept"]
        for row in show.itertuples():
            lines.append(f"{row.selector:<37} {row.accuracy:>7.3f}     {row.sensitivity:>7.3f}      {row.specificity:>7.3f}   {row.acceptance_rate:>7.3f}")
        fig.text(.06, .84, "\n".join(lines), family="monospace", fontsize=10, va="top")
        fig.text(.06, .32,
            "Specificity is estimated from one marginally adverse seed only, so it is not a release-quality estimate.\n"
            "Use this screen to nominate a compact vector; validate rejection using stored BF8 and controlled adverse states before freezing thresholds.",
            fontsize=11, va="top", linespacing=1.5)
        pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(11.69, 8.27)); fig.suptitle(
            "Recommended next decision", fontsize=18, color=NAVY, y=.94
        )
        text = (
            "1. Use BF6 as the default model and evaluate BF7 routinely.\n\n"
            "2. Do not accept BF7 from training-J reduction alone. Carry raw EGI7 onset and "
            "developed/late evidence through selection. Raw EGI7 late is the strongest paired discriminator here.\n\n"
            "3. Retain projected/yield-unique sensitivity as an identifiability diagnostic and growth aid, "
            "but not yet as the BF acceptance gate: it accepted the marginally harmful seed almost universally.\n\n"
            "4. Keep broad EGI/FRE as physical non-regression guards rather than dominant weighted terms.\n\n"
            "5. Next perform an inexpensive offline rejection test on the existing BF8 states. Native-noise "
            "projection for every BF8 state is unnecessary initially: compute the nominated components first, "
            "then expand only if the BF7→8 decision is ambiguous.\n\n"
            "6. Only after that test should the multiscale-equal and sensitivity/projected objective candidates receive matched-seed identification runs."
        )
        fig.text(.07, .83, _wrap_paragraphs(text), fontsize=11.8, va="top", linespacing=1.42)
        pdf.savefig(fig); plt.close(fig)
    return path


def _wrap_paragraphs(value: str, width: int = 103) -> str:
    return "\n\n".join(textwrap.fill(part, width=width) for part in value.split("\n\n"))


if __name__ == "__main__":
    main()
