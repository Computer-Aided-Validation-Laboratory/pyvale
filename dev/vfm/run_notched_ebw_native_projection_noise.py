"""Run a resumable native-DOF projection study with calibrated strain noise.

The study is offline: known maps are used only for evaluation, and production
identification/objective code is not modified.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

from pyvale.vfm import (
    EquilibriumGapMetric, ExperimentData, SliceConfig,
    SliceWiseForceReconstructionMetric, load_identification_result,
)
from pyvale.vfm.postprocessing import (
    evaluate_snapshot_parameter_maps, load_constitutive_law_from_result,
    load_known_parameter_maps,
)
from pyvale.vfm.spatialparambasisfuncs import _compute_variance_range

from analyse_notched_ebw_component_library import REGIMES
from analyse_notched_ebw_gate_campaign import (
    _active_masks, _basis_count, _complete_maps,
)
from analyse_notched_ebw_sensitivity_information import (
    _information_scores, _weights,
)


YIELD_RANGE = 1800.0
HARDENING_RANGE = 9500.0


@dataclass(slots=True)
class NativeState:
    name: str
    seed: int
    basis_count: int
    maps: dict[str, np.ndarray]
    snapshot: object


def main() -> None:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset.expanduser().resolve()
    campaign = args.campaign_root.expanduser().resolve()
    noise_model = json.loads(args.noise_model.read_text(encoding="utf-8"))
    experiment = ExperimentData.load_from_file(
        dataset / "prepared" / "experiment_data.yaml"
    )
    known_raw = load_known_parameter_maps(
        dataset / "prepared" / "known_parameter_maps.npz"
    )
    if known_raw is None:
        raise RuntimeError("Known maps are required for offline evaluation.")
    known = {key: np.asarray(value, float) for key, value in known_raw.items()}
    result_path = next(iter(sorted(campaign.glob("*/identification_result.yaml"))))
    law = load_constitutive_law_from_result(load_identification_result(result_path))
    windows = tuple(int(value) for value in args.windows.split(","))
    scales = tuple(float(value) for value in args.noise_scales.split(","))
    metrics = _create_metrics(experiment, windows)
    mask = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(
        experiment.specimen_geometry.x, experiment.specimen_geometry.y
    )
    yielded, high_plastic = _active_masks(experiment, law, known, mask)
    states = _states(campaign, experiment, known, args.minimum_bf)
    if args.state_index is not None:
        if not 0 <= args.state_index < len(states):
            raise ValueError(
                f"state-index {args.state_index} outside 0..{len(states)-1}"
            )
        states = [states[args.state_index]]
    if args.max_states:
        states = states[:args.max_states]
    checkpoint = output / "projection_noise_rows.jsonl"
    completed = _completed_states(checkpoint) if args.resume else set()
    start = time.monotonic()
    for index, state in enumerate(states, start=1):
        if state.name in completed:
            continue
        rows = _score_state(
            state, experiment, law, metrics, windows, noise_model, scales,
            args.noise_replicates, args.sensitivity_step, known, mask,
            yielded, high_plastic,
        )
        with checkpoint.open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        elapsed = time.monotonic() - start
        print(
            f"[{index:2d}/{len(states)}] {state.name} rows={len(rows)} "
            f"elapsed={elapsed/60:.1f} min", flush=True,
        )
    rows = _load_rows(checkpoint)
    if args.skip_analysis:
        print(json.dumps({
            "output": str(output),
            "states": len({row["state"] for row in rows}),
            "rows": len(rows), "checkpoint": str(checkpoint),
        }, indent=2))
        return
    _write_csv(output / "projection_noise_scores.csv", rows)
    summary_rows = _summaries(rows)
    _write_csv(output / "projection_noise_summary.csv", summary_rows)
    report = _report(output, windows, scales, args, noise_model, rows, summary_rows)
    print(json.dumps({
        "output": str(output), "states": len({row["state"] for row in rows}),
        "rows": len(rows), "report": str(report),
    }, indent=2))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--noise-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", default="7,15,29,57")
    parser.add_argument("--noise-scales", default="0,0.5,1,1.5")
    parser.add_argument("--noise-replicates", type=int, default=64)
    parser.add_argument("--sensitivity-step", type=float, default=0.01)
    parser.add_argument("--minimum-bf", type=int, default=5)
    parser.add_argument("--max-states", type=int, default=0)
    parser.add_argument("--state-index", type=int)
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _create_metrics(experiment, windows):
    egi = [EquilibriumGapMetric(window_size=(size, size)) for size in windows]
    force = SliceWiseForceReconstructionMetric(
        slice_config=SliceConfig(axis="x", num_slices=63)
    )
    for metric in [force, *egi]:
        metric.initialise(experiment)
    return force, egi


def _states(campaign, experiment, known, minimum_bf):
    states = []
    pattern = "spd_sensitivity_gate0p0pct_seed*/identification_result.yaml"
    for path in sorted(campaign.glob(pattern)):
        result = load_identification_result(path)
        seed = int(path.parent.name.rsplit("seed", 1)[1])
        for solve in result.history.phases[-1].solve_results:
            if not solve.accepted or solve.final_snapshot is None:
                continue
            count = _basis_count(solve.final_snapshot)
            if count < minimum_bf:
                continue
            maps = _complete_maps(
                evaluate_snapshot_parameter_maps(solve.final_snapshot, experiment),
                known,
            )
            states.append(NativeState(
                f"seed{seed:02d}_bf{count}", seed, count, maps,
                solve.final_snapshot,
            ))
    return states


def _residual_blocks(stress, experiment, metrics, windows):
    force, egi = metrics
    blocks = {}
    for size, metric in zip(windows, egi, strict=True):
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
    metadata = result.metric_result.additional_fields
    values = np.asarray(metadata["normalised_residual"], float)
    temporal = np.asarray(metadata["temporal_weights"], float)
    spatial = np.asarray(metadata["spatial_weights"], float)
    for regime, indices in REGIMES.items():
        block = values[indices]
        blocks[f"fre__{regime}"] = (
            block, _weights(block, temporal[indices], spatial)
        )
    return blocks


def _score_state(state, experiment, law, metrics, windows, noise_model,
                 scales, replicates, step, known, mask, yielded, high_plastic):
    base_stress = law.calculate_stress(experiment.strain, state.maps)
    base = _residual_blocks(base_stress, experiment, metrics, windows)
    directions = _native_map_directions(state, experiment, step)
    derivatives = {name: [] for name in base}
    groups = []
    for group, changed_maps in directions:
        blocks = _residual_blocks(
            law.calculate_stress(experiment.strain, changed_maps),
            experiment, metrics, windows,
        )
        for name in base:
            derivatives[name].append((blocks[name][0]-base[name][0])/step)
        groups.append(group)

    rng = np.random.default_rng(10_000 + state.seed*100 + state.basis_count)
    unit_noise = [_noise_field(experiment, noise_model, rng)
                  for _ in range(replicates)]
    noisy_blocks = {scale: [] for scale in scales if scale > 0.0}
    one_scale_fields = []
    for noise in unit_noise:
        for scale in noisy_blocks:
            noisy_experiment = ExperimentData(
                experiment.strain + scale*noise,
                experiment.specimen_geometry,
                experiment.boundary_conditions,
                experiment.timesteps,
            )
            fields = _residual_blocks(
                law.calculate_stress(noisy_experiment.strain, state.maps),
                noisy_experiment, metrics, windows,
            )
            noisy_blocks[scale].append(fields)
            if scale == 1.0:
                one_scale_fields.append(fields)
    if not one_scale_fields:
        reference_scale = min(noisy_blocks, key=lambda value: abs(value-1.0))
        one_scale_fields = noisy_blocks[reference_scale]
    sigma = _noise_sigmas(one_scale_fields, base)
    errors = {
        "yielded_rmse_mpa": _rmse(
            state.maps["yield_strength"]-known["yield_strength"], yielded
        ),
        "high_plastic_rmse_mpa": _rmse(
            state.maps["yield_strength"]-known["yield_strength"], high_plastic
        ),
    }
    rows = []
    clean_fields = {name: value for name, value in base.items()}
    all_fields = {0.0: [clean_fields]}
    all_fields.update(noisy_blocks)
    for scale, repetitions in all_fields.items():
        for replicate, fields in enumerate(repetitions):
            for block, (residual, weights) in fields.items():
                whiten = sigma[block]
                scores = _information_scores(
                    residual/whiten, weights,
                    [field/whiten for field in derivatives[block]], groups,
                )
                rows.append({
                    "state": state.name, "seed": state.seed,
                    "basis_count": state.basis_count,
                    "noise_scale": scale, "noise_replicate": replicate,
                    "block": block, **errors,
                    "raw_rms": scores["raw_rms"],
                    "projected_rms": scores["projected_rms"],
                    "yield_unique_rms": scores["yield_unique_rms"],
                    "hardening_unique_rms": scores["hardening_unique_rms"],
                    "native_dofs": len(directions),
                })
    return rows


def _native_map_directions(state, experiment, step):
    x = np.asarray(experiment.specimen_geometry.x, float)
    y = np.asarray(experiment.specimen_geometry.y, float)
    snapshot = state.snapshot.spatial_parameterisations
    basis_summary = next(
        item.summary for item in snapshot["yield_strength"]
        if item.summary.get("kind") == "basis_functions"
    )
    directions = []
    maps = state.maps

    def changed_yield(delta):
        changed = {key: value.copy() for key, value in maps.items()}
        changed["yield_strength"] = np.clip(
            maps["yield_strength"] + delta, 200.0, 2000.0
        )
        return changed

    directions.append(("yield", changed_yield(step*YIELD_RANGE*np.ones_like(x))))
    variance_min, variance_max = _compute_variance_range(x, y)
    half_span = .5*float(np.log(variance_max/variance_min))
    bounds = {
        "height": 2*YIELD_RANGE, "x": float(np.nanmax(x)-np.nanmin(x)),
        "y": float(np.nanmax(y)-np.nanmin(y)),
        "s11": 2*half_span, "s12": 2*half_span, "s22": 2*half_span,
    }
    base_basis = _basis_map(basis_summary, x, y)
    for kernel_index, kernel in enumerate(basis_summary["kernels"]):
        for field in ("height", "x", "y", "s11", "s12", "s22"):
            perturbed = _basis_map(
                basis_summary, x, y,
                override=(kernel_index, field, step*bounds[field]),
            )
            directions.append(("yield", changed_yield(perturbed-base_basis)))
    hardening = {key: value.copy() for key, value in maps.items()}
    hardening["hardening_modulus"] = np.clip(
        maps["hardening_modulus"] + step*HARDENING_RANGE,
        500.0, 10000.0,
    )
    directions.append(("hardening", hardening))
    return directions


def _basis_map(summary, x, y, override=None):
    output = np.zeros_like(x)
    for index, original in enumerate(summary["kernels"]):
        kernel = dict(original)
        centre = list(original["centre"])
        log_covariance = np.asarray(original["log_covariance"], float).copy()
        height = float(original["height"])
        if override is not None and override[0] == index:
            _, field, delta = override
            if field == "height": height += delta
            elif field == "x": centre[0] += delta
            elif field == "y": centre[1] += delta
            elif field == "s11": log_covariance[0, 0] += delta
            elif field == "s12":
                log_covariance[0, 1] += delta; log_covariance[1, 0] += delta
            elif field == "s22": log_covariance[1, 1] += delta
        values, vectors = np.linalg.eigh(log_covariance)
        covariance = float(original["reference_variance"]) * (
            (vectors*np.exp(values)) @ vectors.T
        )
        inverse = np.linalg.inv(covariance)
        dx = x-centre[0]; dy = y-centre[1]
        exponent = inverse[0, 0]*dx*dx + 2*inverse[0, 1]*dx*dy + inverse[1, 1]*dy*dy
        output += height*np.exp(-.5*exponent)
    return output


def _noise_field(experiment, model, rng):
    shape = experiment.strain.shape
    dx = float(np.nanmedian(np.abs(np.diff(experiment.specimen_geometry.x, axis=1))))
    dy = float(np.nanmedian(np.abs(np.diff(experiment.specimen_geometry.y, axis=0))))
    output = np.empty(shape, dtype=float)
    for component, name in enumerate(("exx", "eyy", "exy")):
        config = model["components"][name]
        sx = max(float(config["gaussian_filter_sigma_mm"]["x"])/dx, .05)
        sy = max(float(config["gaussian_filter_sigma_mm"]["y"])/dy, .05)
        for timestep in range(shape[0]):
            field = gaussian_filter(
                rng.standard_normal(shape[2:]), sigma=(sy, sx), mode="reflect"
            )
            field *= float(config["sigma"])/max(float(np.std(field)), 1e-15)
            output[timestep, component] = field
    return output


def _noise_sigmas(repetitions, base):
    output = {}
    for block in base:
        stack = np.stack([fields[block][0] for fields in repetitions])
        valid = np.isfinite(base[block][0])
        sigma = np.full(base[block][0].shape, np.nan)
        sigma[valid] = np.std(stack[:, valid], axis=0, ddof=1)
        finite = sigma[np.isfinite(sigma) & (sigma > 0)]
        floor = max(float(np.nanmedian(finite))*.1, 1e-12)
        output[block] = np.where(np.isfinite(sigma), np.maximum(sigma, floor), floor)
    return output


def _completed_states(path):
    return {row["state"] for row in _load_rows(path)} if path.is_file() else set()


def _load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _summaries(rows):
    output = []
    components = ("raw_rms", "projected_rms", "yield_unique_rms", "hardening_unique_rms")
    keys = sorted({(float(r["noise_scale"]), int(r["noise_replicate"]), r["block"])
                   for r in rows})
    for scale, replicate, block in keys:
        selected = [r for r in rows if float(r["noise_scale"]) == scale
                    and int(r["noise_replicate"]) == replicate and r["block"] == block]
        if len(selected) < 3:
            continue
        for target in ("yielded_rmse_mpa", "high_plastic_rmse_mpa"):
            truth = np.asarray([r[target] for r in selected], float)
            for component in components:
                score = np.asarray([r[component] for r in selected], float)
                rho = float(spearmanr(score, truth).statistic)
                output.append({
                    "noise_scale": scale, "noise_replicate": replicate,
                    "block": block, "target": target, "component": component,
                    "spearman_r": rho,
                    "pairwise_accuracy": _pairwise(score, truth),
                })
    return output


def _pairwise(score, truth):
    ds = score[:, None]-score[None, :]; dt = truth[:, None]-truth[None, :]
    use = np.triu(np.ones(ds.shape, bool), 1) & (np.abs(ds) > 1e-12) & (np.abs(dt) > 1e-12)
    return float(np.mean(np.sign(ds[use]) == np.sign(dt[use]))) if np.any(use) else float("nan")


def _report(output, windows, scales, args, noise_model, rows, summary):
    report = output / "NOTCHED_EBW_NATIVE_PROJECTION_NOISE.pdf"
    aggregated = {}
    for target in ("yielded_rmse_mpa", "high_plastic_rmse_mpa"):
        candidates = {}
        for row in summary:
            if row["target"] != target:
                continue
            key = (float(row["noise_scale"]), row["block"], row["component"])
            candidates.setdefault(key, []).append(float(row["spearman_r"]))
        aggregated[target] = sorted(
            ({"scale": key[0], "block": key[1], "component": key[2],
              "mean": float(np.nanmean(values)), "std": float(np.nanstd(values))}
             for key, values in candidates.items()),
            key=lambda row: row["mean"], reverse=True,
        )
    with PdfPages(report) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27)); fig.suptitle(
            "Native-DOF projected residual under WDBN1-calibrated noise", fontsize=18, y=.94)
        text = (f"States: {len({r['state'] for r in rows})}; windows: {windows}\n"
                f"Noise scales: {scales}; replicates: {args.noise_replicates}\n"
                f"WDBN1 strain sigma (microstrain): " + ", ".join(
                    f"{k}={v['sigma_microstrain']:.1f}" for k,v in noise_model['components'].items()) +
                "\n\nProjection uses each state's active homogeneous yield, Gaussian centre,\n"
                "height and SPD covariance DOFs, plus homogeneous hardening.\n"
                "Noise whitening is diagonal after propagation through each metric.\n"
                "Synthetic truth is used only for ranking evaluation.")
        fig.text(.07, .82, text, fontsize=12, va="top", linespacing=1.55); pdf.savefig(fig); plt.close(fig)
        for target in aggregated:
            fig = plt.figure(figsize=(11.69, 8.27)); fig.suptitle(target.replace("_", " ").title(), fontsize=17, y=.94)
            lines=[]
            one=[r for r in aggregated[target] if r["scale"] == 1.0][:18]
            for r in one: lines.append(f"{r['block']:<34} {r['component']:<22} rho={r['mean']:+.3f} +/- {r['std']:.3f}")
            fig.text(.06,.87,"\n".join(lines),family="monospace",fontsize=9,va="top"); pdf.savefig(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(11.69, 8.27), constrained_layout=True)
        for component in ("raw_rms", "projected_rms", "yield_unique_rms"):
            values=[]
            for scale in scales:
                candidates=[r for r in aggregated["high_plastic_rmse_mpa"] if r["scale"]==scale and r["component"]==component]
                values.append(max((r["mean"] for r in candidates), default=np.nan))
            ax.plot(scales, values, marker="o", label=component)
        ax.set(xlabel="WDBN1 noise scale", ylabel="Best mean Spearman rho", ylim=(-1,1), title="High-plastic discrimination versus noise")
        ax.legend(); ax.grid(alpha=.25); pdf.savefig(fig); plt.close(fig)
    return report


def _write_csv(path, rows):
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _rmse(error, selection):
    return float(np.sqrt(np.mean(np.asarray(error)[selection]**2)))


if __name__ == "__main__":
    main()
