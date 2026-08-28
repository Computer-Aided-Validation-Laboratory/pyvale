"""Offline sensitivity-information study for the notched-EBW campaign.

This script does not alter the production objective or identification code.
Synthetic truth is used only after scoring, to evaluate discrimination.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.stats import spearmanr

from pyvale.vfm import (
    EquilibriumGapMetric,
    ExperimentData,
    SliceConfig,
    SliceWiseForceReconstructionMetric,
    load_identification_result,
)
from pyvale.vfm.postprocessing import (
    load_constitutive_law_from_result,
    load_known_parameter_maps,
)

from analyse_notched_ebw_component_library import (
    REGIMES,
    State,
    _optimiser_late_states,
)
from analyse_notched_ebw_gate_campaign import _active_masks, _complete_maps


WINDOWS = (15, 29, 57)
YIELD_RANGE = 1800.0
HARDENING_RANGE = 9500.0
STEP = float(os.environ.get("PYVALE_SENSITIVITY_INFORMATION_STEP", "0.01"))
DIAGNOSTICS = (
    "raw_rms",
    "sensitivity_rms",
    "leverage_rms",
    "projected_rms",
    "yield_unique_rms",
    "hardening_unique_rms",
)


def main() -> None:
    args = _parse_args()
    dataset = args.dataset.expanduser().resolve()
    campaign = args.campaign_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    experiment = ExperimentData.load_from_file(
        dataset / "prepared" / "experiment_data.yaml"
    )
    known_raw = load_known_parameter_maps(
        dataset / "prepared" / "known_parameter_maps.npz"
    )
    if known_raw is None:
        raise RuntimeError("Known maps are required for offline evaluation.")
    known = {k: np.asarray(v, dtype=float) for k, v in known_raw.items()}
    result_path = next(iter(sorted(campaign.glob("*/identification_result.yaml"))))
    law = load_constitutive_law_from_result(
        load_identification_result(result_path)
    )
    mask = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(
        experiment.specimen_geometry.x, experiment.specimen_geometry.y
    )
    yielded, high_plastic = _active_masks(experiment, law, known, mask)
    metrics = _create_metrics(experiment)
    directions = _directions(experiment, mask)
    states = _load_states(args.component_analysis, campaign, experiment, known)
    if args.state_source != "all":
        states = [state for state in states if state.source == args.state_source]

    checkpoint = output / "state_information.jsonl"
    completed = _load_checkpoint(checkpoint) if args.resume else {}
    total = min(len(states), args.max_states) if args.max_states else len(states)
    rows = list(completed.values())
    done_names = set(completed)
    selected = [state for state in states if state.name not in done_names][:total]
    for index, state in enumerate(selected, start=len(done_names) + 1):
        row = _score_state(
            state, law, experiment, metrics, directions, known, mask,
            yielded, high_plastic,
        )
        with checkpoint.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        rows.append(row)
        print(f"[{index:3d}/{total}] {state.name}", flush=True)

    _write_csv(output / "state_information.csv", rows)
    discrimination = _discrimination(rows)
    _write_csv(output / "information_discrimination.csv", discrimination)
    transitions = _transition_analysis(rows)
    _write_csv(output / "optimiser_transition_discrimination.csv", transitions)
    summary = _summarise(discrimination, transitions, rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_pdf(output / "NOTCHED_EBW_SENSITIVITY_INFORMATION.pdf", summary,
               discrimination, transitions, rows)
    print(json.dumps({
        "output": str(output),
        "states": len(rows),
        "directions": [item[0] for item in directions],
        "report": str(output / "NOTCHED_EBW_SENSITIVITY_INFORMATION.pdf"),
    }, indent=2))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--component-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-states", type=int, default=0)
    parser.add_argument(
        "--state-source", choices=("all", "independent", "optimiser_late"),
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _create_metrics(experiment):
    egi = [EquilibriumGapMetric(window_size=(size, size)) for size in WINDOWS]
    force = SliceWiseForceReconstructionMetric(
        slice_config=SliceConfig(axis="x", num_slices=63)
    )
    for metric in [force, *egi]:
        metric.initialise(experiment)
    return force, egi


def _load_states(component_analysis, campaign, experiment, known):
    archive = np.load(Path(component_analysis) / "independent_state_maps.npz")
    states = []
    for index, name in enumerate(archive["names"].astype(str)):
        maps = {key: value.copy() for key, value in known.items()}
        maps["yield_strength"] = archive["yield_strength"][index].copy()
        maps["hardening_modulus"] = archive["hardening_modulus"][index].copy()
        states.append(State(
            name=name, source="independent",
            split=str(archive["splits"][index]),
            family=str(archive["families"][index]), parameters={}, maps=maps,
        ))
    states.extend(_optimiser_late_states(campaign, experiment, known))
    return states


def _directions(experiment, mask):
    x = np.asarray(experiment.specimen_geometry.x, dtype=float)
    y = np.asarray(experiment.specimen_geometry.y, dtype=float)
    xmin, xmax = float(np.min(x[mask])), float(np.max(x[mask]))
    ymin, ymax = float(np.min(y[mask])), float(np.max(y[mask]))
    centres_x = np.linspace(xmin + .25 * (xmax-xmin), xmax - .25*(xmax-xmin), 3)
    centres_y = np.linspace(ymin + .22 * (ymax-ymin), ymax - .22*(ymax-ymin), 3)
    sx, sy = (xmax-xmin)/9.0, (ymax-ymin)/5.0
    output = [("yield_global", "yield", np.ones_like(x))]
    for iy, cy in enumerate(centres_y):
        for ix, cx in enumerate(centres_x):
            field = np.exp(-0.5 * (((x-cx)/sx)**2 + ((y-cy)/sy)**2))
            field[~mask] = 0.0
            output.append((f"yield_g{iy}{ix}", "yield", field))
    output.append(("hardening_global", "hardening", np.ones_like(x)))
    return output


def _residual_blocks(stress, experiment, metrics):
    force, egi = metrics
    blocks = {}
    for size, metric in zip(WINDOWS, egi, strict=True):
        result = metric.evaluate_equilibrium_gap(stress)
        temporal = np.asarray(
            result.metric_result.additional_fields["temporal_weights"], float
        )
        for regime, indices in REGIMES.items():
            values = np.asarray(result.normalised_gap, float)[indices]
            blocks[f"egi{size}__{regime}"] = (
                values, _weights(values, temporal[indices], None)
            )
    result = force.evaluate_force_recon_error(stress, experiment)
    fields = result.metric_result.additional_fields
    values = np.asarray(fields["normalised_residual"], float)
    temporal = np.asarray(fields["temporal_weights"], float)
    spatial = np.asarray(fields["spatial_weights"], float)
    for regime, indices in REGIMES.items():
        block = values[indices]
        blocks[f"fre__{regime}"] = (
            block, _weights(block, temporal[indices], spatial)
        )
    return blocks


def _weights(values, temporal, spatial):
    time = np.asarray(temporal, float)
    time = time / max(float(np.sum(time)), np.finfo(float).eps)
    if values.ndim == 3:
        weights = np.broadcast_to(time[:, None, None], values.shape).copy()
    else:
        space = np.asarray(spatial, float)
        space = space / max(float(np.sum(space)), np.finfo(float).eps)
        weights = time[:, None] * space[None, :]
    weights[~np.isfinite(values)] = 0.0
    weights /= max(float(np.sum(weights)), np.finfo(float).eps)
    return weights


def _score_state(state, law, experiment, metrics, directions, known, mask,
                 yielded, high_plastic):
    maps = _complete_maps(state.maps, known)
    base_stress = law.calculate_stress(experiment.strain, maps)
    base = _residual_blocks(base_stress, experiment, metrics)
    derivatives = {name: [] for name in base}
    groups = []
    for _, group, field in directions:
        changed = {key: value.copy() for key, value in maps.items()}
        if group == "yield":
            changed["yield_strength"] = np.clip(
                changed["yield_strength"] + STEP * YIELD_RANGE * field,
                200.0, 2000.0,
            )
        else:
            changed["hardening_modulus"] = np.clip(
                changed["hardening_modulus"] + STEP * HARDENING_RANGE * field,
                500.0, 10000.0,
            )
        perturbed = _residual_blocks(
            law.calculate_stress(experiment.strain, changed), experiment, metrics
        )
        for name in base:
            derivatives[name].append((perturbed[name][0] - base[name][0]) / STEP)
        groups.append(group)

    error = maps["yield_strength"] - known["yield_strength"]
    row = {
        "name": state.name, "source": state.source, "split": state.split,
        "family": state.family, "basis_count": state.basis_count,
        "seed": state.seed,
        "yielded_rmse_mpa": _rmse(error, yielded),
        "high_plastic_rmse_mpa": _rmse(error, high_plastic),
    }
    for block, (residual, weights) in base.items():
        values = _information_scores(
            residual, weights, derivatives[block], groups
        )
        for diagnostic, value in values.items():
            row[f"{block}__{diagnostic}"] = value
    return row


def _information_scores(residual, weights, derivative_fields, groups):
    valid = np.isfinite(residual) & (weights > 0.0)
    for field in derivative_fields:
        valid &= np.isfinite(field)
    root_w = np.sqrt(weights[valid])
    r = np.asarray(residual)[valid] * root_w
    s = np.column_stack([np.asarray(field)[valid] * root_w
                         for field in derivative_fields])
    sy = s[:, np.asarray(groups) == "yield"]
    sh = s[:, np.asarray(groups) == "hardening"]
    row_energy = np.sum(s*s, axis=1)
    leverage = _leverage(s)
    sy_unique = _residualise(sy, sh)
    sh_unique = _residualise(sh, sy)
    return {
        "raw_rms": float(np.linalg.norm(r)),
        "sensitivity_rms": _weighted_score(r, row_energy),
        "leverage_rms": _weighted_score(r, leverage),
        "projected_rms": _projection_score(r, s),
        "yield_unique_rms": _projection_score(r, sy_unique),
        "hardening_unique_rms": _projection_score(r, sh_unique),
        "information_trace": float(np.sum(leverage)),
        "yield_unique_information": float(np.sum(sy_unique*sy_unique)),
        "hardening_unique_information": float(np.sum(sh_unique*sh_unique)),
    }


def _orthonormal(matrix):
    if matrix.size == 0:
        return np.empty((matrix.shape[0], 0))
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    keep = singular > max(singular[0] * 1e-8, np.finfo(float).eps)
    return u[:, keep]


def _residualise(target, nuisance):
    q = _orthonormal(nuisance)
    return target - q @ (q.T @ target)


def _leverage(matrix):
    q = _orthonormal(matrix)
    return np.sum(q*q, axis=1)


def _projection_score(residual, matrix):
    q = _orthonormal(matrix)
    if q.shape[1] == 0:
        return 0.0
    return float(np.linalg.norm(q.T @ residual) / np.sqrt(q.shape[1]))


def _weighted_score(residual, weights):
    total = float(np.sum(weights))
    if total <= np.finfo(float).eps:
        return 0.0
    return float(np.sqrt(np.sum(weights * residual*residual) / total))


def _rmse(error, selection):
    return float(np.sqrt(np.mean(np.asarray(error)[selection] ** 2)))


def _load_checkpoint(path):
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {row["name"]: row for row in rows}


def _score_columns(rows):
    return [key for key in rows[0] if any(key.endswith("__" + d) for d in DIAGNOSTICS)]


def _discrimination(rows):
    subsets = {
        "development": [r for r in rows if r["split"] == "development"],
        "validation": [r for r in rows if r["split"] == "validation"],
        "optimiser_bf5_8": [r for r in rows if r["source"] == "optimiser_late"],
    }
    output = []
    for subset, selected in subsets.items():
        if len(selected) < 3:
            continue
        for target in ("yielded_rmse_mpa", "high_plastic_rmse_mpa"):
            truth = np.asarray([r[target] for r in selected])
            for column in _score_columns(rows):
                score = np.asarray([r[column] for r in selected])
                rho = float(spearmanr(score, truth).statistic)
                accuracy, coverage = _pairwise(score, truth)
                parts = column.split("__")
                output.append({
                    "subset": subset, "target": target, "component": column,
                    "metric": parts[0], "regime": parts[1],
                    "diagnostic": parts[2], "spearman_r": rho,
                    "pairwise_accuracy": accuracy, "pairwise_coverage": coverage,
                    "states": len(selected),
                })
    return output


def _pairwise(score, truth):
    ds = score[:, None] - score[None, :]
    dt = truth[:, None] - truth[None, :]
    upper = np.triu(np.ones(ds.shape, bool), 1)
    informative = upper & (np.abs(ds) > 1e-12) & (np.abs(dt) > 1e-12)
    possible = upper & (np.abs(dt) > 1e-12)
    if not np.any(informative):
        return float("nan"), 0.0
    return (float(np.mean(np.sign(ds[informative]) == np.sign(dt[informative]))),
            float(np.sum(informative) / max(np.sum(possible), 1)))


def _transition_analysis(rows):
    selected = [row for row in rows if row["source"] == "optimiser_late"]
    output = []
    for target in ("yielded_rmse_mpa", "high_plastic_rmse_mpa"):
        for column in _score_columns(rows):
            within = []
            for basis_count in (5, 6, 7, 8):
                group = [row for row in selected
                         if int(row["basis_count"]) == basis_count]
                rho = float(spearmanr(
                    [float(row[column]) for row in group],
                    [float(row[target]) for row in group],
                ).statistic)
                if np.isfinite(rho):
                    within.append(rho)
            adjacent = []
            bf7_to_8 = []
            for seed in range(8):
                group = sorted(
                    [row for row in selected if int(row["seed"]) == seed],
                    key=lambda row: int(row["basis_count"]),
                )
                for before, after in zip(group, group[1:]):
                    correct = np.sign(float(after[column])-float(before[column])) == np.sign(float(after[target])-float(before[target]))
                    adjacent.append(correct)
                    if int(before["basis_count"]) == 7:
                        bf7_to_8.append(correct)
            parts = column.split("__")
            output.append({
                "target": target, "component": column, "metric": parts[0],
                "regime": parts[1], "diagnostic": parts[2],
                "mean_within_basis_spearman": float(np.mean(within)) if within else float("nan"),
                "adjacent_bf_accuracy": float(np.mean(adjacent)),
                "bf7_to_8_accuracy": float(np.mean(bf7_to_8)),
            })
    return output


def _summarise(discrimination, transitions, rows):
    best = {}
    for subset in ("development", "validation", "optimiser_bf5_8"):
        best[subset] = {}
        for target in ("yielded_rmse_mpa", "high_plastic_rmse_mpa"):
            candidates = [r for r in discrimination
                          if r["subset"] == subset and r["target"] == target]
            candidates.sort(key=lambda r: (np.nan_to_num(r["spearman_r"], nan=-2.0),
                                            r["pairwise_accuracy"]), reverse=True)
            best[subset][target] = candidates[:10]
    transition_best = {}
    for target in ("yielded_rmse_mpa", "high_plastic_rmse_mpa"):
        candidates = [row for row in transitions if row["target"] == target]
        candidates.sort(key=lambda row: (
            row["mean_within_basis_spearman"],
            row["adjacent_bf_accuracy"], row["bf7_to_8_accuracy"],
        ), reverse=True)
        transition_best[target] = candidates[:10]
    return {"states": len(rows), "windows": list(WINDOWS), "best": best,
            "transition_best": transition_best}


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _write_pdf(path, summary, discrimination, transitions, rows):
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27)); fig.suptitle(
            "Notched-EBW sensitivity-information pilot", fontsize=20, y=.94)
        text = (f"Offline states: {len(rows)}\nEGI windows: 15, 29, 57; plus FRE\n"
                "Diagnostics: raw, sensitivity magnitude, leverage, full projection,\n"
                "yield-unique projection and hardening-unique projection.\n\n"
                "Truth maps were used only to evaluate ranking after scoring.\n"
                "Production objective and identification code were not changed.")
        fig.text(.09, .76, text, fontsize=13, va="top", linespacing=1.6); pdf.savefig(fig); plt.close(fig)
        for target, title in (("yielded_rmse_mpa", "Yielded-region error"),
                              ("high_plastic_rmse_mpa", "High-plastic-region error")):
            fig, axes = plt.subplots(1, 3, figsize=(11.69, 8.27), constrained_layout=True)
            for ax, subset in zip(axes, ("development", "validation", "optimiser_bf5_8"), strict=True):
                data = [r for r in discrimination if r["subset"] == subset and r["target"] == target]
                labels = sorted(set(r["diagnostic"] for r in data))
                values = [[r["spearman_r"] for r in data if r["diagnostic"] == d] for d in labels]
                if values:
                    ax.boxplot(values, tick_labels=[d.replace("_rms", "") for d in labels], orientation="horizontal")
                else:
                    ax.text(.5, .5, "Insufficient states", ha="center", va="center",
                            transform=ax.transAxes)
                ax.axvline(0, color="0.5", lw=1); ax.set_xlim(-1, 1); ax.set_title(subset.replace("_", " "))
                ax.set_xlabel("Spearman rank correlation")
            fig.suptitle(title, fontsize=16); pdf.savefig(fig); plt.close(fig)
        fig = plt.figure(figsize=(11.69, 8.27)); fig.suptitle("Strongest held-out evidence", fontsize=18, y=.94)
        lines = []
        for target in ("yielded_rmse_mpa", "high_plastic_rmse_mpa"):
            lines.append(target.replace("_", " ").upper())
            for row in summary["best"].get("validation", {}).get(target, [])[:6]:
                lines.append(f"  {row['component']}: rho={row['spearman_r']:.3f}, pairwise={row['pairwise_accuracy']:.3f}")
            lines.append("")
        fig.text(.06, .86, "\n".join(lines), family="monospace", fontsize=10, va="top")
        pdf.savefig(fig); plt.close(fig)
        fig = plt.figure(figsize=(11.69, 8.27)); fig.suptitle(
            "Late-stage optimiser discrimination", fontsize=18, y=.94)
        lines = ["Ranked primarily by mean within-BF Spearman correlation;",
                 "adjacent and BF7->8 columns test actual growth decisions.", ""]
        for target in ("yielded_rmse_mpa", "high_plastic_rmse_mpa"):
            lines.append(target.replace("_", " ").upper())
            for row in summary["transition_best"][target][:7]:
                lines.append(
                    f"  {row['component']:<55} within={row['mean_within_basis_spearman']:.3f} "
                    f"adj={row['adjacent_bf_accuracy']:.3f} 7->8={row['bf7_to_8_accuracy']:.3f}"
                )
            lines.append("")
        fig.text(.035, .87, "\n".join(lines), family="monospace", fontsize=8.7, va="top")
        pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
