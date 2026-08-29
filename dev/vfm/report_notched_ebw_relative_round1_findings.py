"""Report the corrected relative-regime notched-EBW Round-1 findings."""

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
    screen = root / "screen"
    configuration = _json(screen / "screen_configuration.json")
    manifest = _json(screen / "screen_manifest.json")
    selected = _json(screen / "selected_objectives.json")
    floors = _json(screen / "noise_floors.json")
    states = _csv(screen / "state_feature_rows.csv")
    information = _csv(screen / "window_information_scores.csv")
    direct = _csv(root / "direct_fit/direct_fit_reference.csv")

    rich = selected["raw_information_rich"]
    simple = selected["raw_parsimonious"]
    ranking = selected["ranking"]
    truth = next(row for row in states if row["name"] == "reference_truth")
    feature_keys = {
        "Fine EGI onset CVaR90": "egi_1.4mm__onset__cvar90",
        "Middle EGI late RMS": "egi_5.8mm__late__rms",
        "Broad EGI onset CVaR90": "egi_11.4mm__onset__cvar90",
        "FRE late coherent RMS": "fre__late__coherent_rms",
    }
    ratios = {
        label: float(truth[key]) / float(floors[key])
        for label, key in feature_keys.items()
    }

    with PdfPages(output) as pdf:
        _summary(pdf, manifest, configuration, rich, simple, ranking)
        _offline_evidence(pdf, ranking, information, rich, simple)
        _late_diagnostics(pdf, information, rich, simple, ratios)
        _capacity_and_decision(pdf, direct)
    print(f"relative Round-1 findings={output}", flush=True)


def _summary(pdf, manifest, configuration, rich, simple, ranking):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle("Notched-EBW corrected Round 1 findings", fontsize=19, y=.95)
    regimes = configuration["load_regimes"]
    blocks = ", ".join(
        f"{name.replace('_', ' ')} {values[0]}-{values[-1]}"
        for name, values in (
            ("pre_yield", regimes["pre_yield"]), ("onset", regimes["onset"]),
            ("developed", regimes["developed"]), ("late", regimes["late"]),
        )
    )
    lines = [
        "Decision: advance to the targeted 16-case raw-objective online pilot. "
        "Do not yet launch the projected candidate or full confirmation round.",
        f"Evidence: {manifest['states']} states and {manifest['noise_replicates']} independent WDBN1 noise realisations.",
        f"The repaired Phase-0 partition is disjoint and multi-frame: {blocks}.",
        f"Information-rich candidate: 7/29/57 points, alpha={rich['alpha']}, merit={ranking[0]['merit']:.4f}.",
        f"Parsimonious candidate: 7/57 points, alpha={simple['alpha']}, merit={next(x['merit'] for x in ranking if x['name']==simple['name']):.4f}.",
        f"The rich candidate improves offline merit by {ranking[0]['merit']-next(x['merit'] for x in ranking if x['name']==simple['name']):.4f}, but alpha=0.50 versus 0.75 differs by only {ranking[0]['merit']-ranking[1]['merit']:.1e}.",
        "Therefore alpha=0.50 is a pragmatic frozen setting, not a statistically unique optimum.",
    ]
    y = .84
    for line in lines:
        fig.text(.07, y, textwrap.fill(line, 108), fontsize=11.3, va="top")
        y -= .095
    fig.text(.07, .12, "What Round 1 establishes", fontsize=14, weight="bold")
    fig.text(.07, .07, "The repaired features rank broad independent perturbations well enough to justify an online test. They do not establish that the optimiser will follow the map-error gradient at late BF counts.", fontsize=11, wrap=True)
    pdf.savefig(fig); plt.close(fig)


def _offline_evidence(pdf, ranking, information, rich, simple):
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), constrained_layout=True)
    names = [row["name"].replace("raw_", "") for row in ranking]
    merits = np.asarray([float(row["merit"]) for row in ranking])
    axes[0].barh(np.arange(len(names)), merits, color=["tab:blue" if i < 3 else "0.6" for i in range(len(names))])
    axes[0].set_yticks(np.arange(len(names)), names); axes[0].invert_yaxis()
    axes[0].set_xlim(merits.min() - .003, merits.max() + .001)
    axes[0].set(xlabel="Held-out ranking merit", title="Three-window 5.8 mm family leads")
    axes[0].grid(axis="x", alpha=.25)

    rows = []
    for candidate, label in ((rich["name"], "7/29/57"), (simple["name"], "7/57")):
        for split in ("development", "validation"):
            for target, short in (("yielded_rmse_mpa", "Yielded"), ("high_plastic_rmse_mpa", "High plastic")):
                row = next(item for item in information if item["candidate"] == candidate and item["subset"] == split and item["target"] == target)
                rows.append((label, split, short, float(row["spearman_r"])))
    x = np.arange(4); width=.34
    labels = ["Dev\nyielded", "Dev\nhigh plastic", "Val\nyielded", "Val\nhigh plastic"]
    for index, candidate in enumerate(("7/29/57", "7/57")):
        values = [row[3] for row in rows if row[0] == candidate]
        axes[1].bar(x + (index-.5)*width, values, width, label=candidate)
    axes[1].set_xticks(x, labels); axes[1].set_ylim(.9, 1.0)
    axes[1].set(ylabel="Spearman rho", title="Independent-family ranking is strong")
    axes[1].grid(axis="y", alpha=.25); axes[1].legend()
    fig.suptitle("Offline selection evidence", fontsize=16)
    pdf.savefig(fig); plt.close(fig)


def _late_diagnostics(pdf, information, rich, simple, ratios):
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), constrained_layout=True)
    for candidate, label in ((rich["name"], "7/29/57"), (simple["name"], "7/57")):
        rows = sorted(
            (row for row in information if row["candidate"] == candidate and row["subset"].startswith("diagnostic_bf") and row["target"] == "high_plastic_rmse_mpa"),
            key=lambda row: int(row["subset"].split("bf")[1]),
        )
        axes[0].plot(
            [int(row["subset"].split("bf")[1]) for row in rows],
            [float(row["spearman_r"]) for row in rows], marker="o", label=label,
        )
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(xlabel="BF count", ylabel="Within-BF Spearman rho", ylim=(-1, 1), title="Late optimiser-state discrimination remains weak")
    axes[0].grid(alpha=.25); axes[0].legend()
    labels = list(ratios); values = [ratios[label] for label in labels]
    axes[1].barh(np.arange(len(labels)), values, color=["tab:red" if value <= 1 else "tab:orange" for value in values])
    axes[1].axvline(1, color="black", linewidth=1, label="truth = noise floor")
    axes[1].set_yticks(np.arange(len(labels)), labels); axes[1].invert_yaxis()
    axes[1].set(xlabel="Truth residual / propagated noise floor", title="Selected features sit near the noise floor")
    axes[1].grid(axis="x", alpha=.25); axes[1].legend()
    fig.suptitle("Why an online pilot is still required", fontsize=16)
    fig.text(.5, .015, "Broad perturbation ranking is not equivalent to local optimisation guidance. BF7-BF8 behaviour and small signal margins are the main risks to test online.", ha="center", fontsize=9.8)
    pdf.savefig(fig); plt.close(fig)


def _capacity_and_decision(pdf, direct):
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), constrained_layout=True)
    bf = [int(row["basis_count"]) for row in direct]
    for key, label in (("roi_rmse_mpa", "ROI"), ("yielded_rmse_mpa", "Yielded"), ("high_plastic_rmse_mpa", "High plastic")):
        axes[0].plot(bf, [float(row[key]) for row in direct], marker="o", label=label)
    axes[0].set(xlabel="SPD Gaussian BF count", ylabel="Best direct-fit RMSE [MPa]", title="Representability target")
    axes[0].grid(alpha=.25); axes[0].legend()
    axes[1].axis("off")
    steps = [
        "1. Run current 29/57, multiscale-equal 7/29/57, raw 7/57, and raw 7/29/57.",
        "2. Use two matched seeds, clean and 1x WDBN1 noise: 16 cases, BF7 cap.",
        "3. Compare each BF trajectory with the direct-fit curve: recovery-gap AUC, yielded/high-plastic errors, closure and seed stability.",
        "4. Advance a hybrid only if it beats multiscale equal without degrading force/mechanical closure or robustness.",
        "5. Keep projection gated until native-DOF preparation and BF-refresh tests pass.",
        "6. Replace hard temporal blocks later with frozen continuous sensitivity weights if the pilot supports the feature family.",
    ]
    axes[1].text(.02, .96, "Recommended next step", fontsize=15, weight="bold", va="top")
    y=.86
    for step in steps:
        axes[1].text(.03, y, textwrap.fill(step, 58), fontsize=11, va="top")
        y -= .135
    fig.suptitle("Online decision gate", fontsize=16)
    pdf.savefig(fig); plt.close(fig)


def _csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
