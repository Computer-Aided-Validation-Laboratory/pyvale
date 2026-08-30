"""Evaluate offline scalar reductions of notched-EBW EGI/FRE fields.

The true synthetic material map is used only to score how well each candidate
scalar ranks stored material-map states.  Candidate objectives themselves use
only quantities available during identification.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
from statistics import median

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import spearmanr

from pyvale.vfm import ExperimentData, load_identification_result
from pyvale.vfm.postprocessing import (
    compute_plasticity_diagnostics,
    evaluate_snapshot_parameter_maps,
    load_constitutive_law_from_result,
    load_known_parameter_maps,
)

from analyse_notched_ebw_gate_campaign import (
    FORCE_WEIGHT,
    WINDOW_WEIGHTS,
    _active_masks,
    _basis_count,
    _complete_maps,
    _completed_runs,
    _controlled_states,
    _create_metrics,
    _extract_baselines,
    _objective_from_scalars,
    _residual_blocks,
)


COMBINATION_WEIGHTS = np.asarray(
    [
        (1.0 - FORCE_WEIGHT) * WINDOW_WEIGHTS[0],
        (1.0 - FORCE_WEIGHT) * WINDOW_WEIGHTS[1],
        FORCE_WEIGHT,
    ],
    dtype=np.float64,
)
AGGREGATIONS = (
    "rms",
    "hierarchical_mean",
    "time_p90",
    "tail_p90",
    "tail_p95",
    "coherent_rms",
    "plastic_active_rms",
)
COMBINERS = ("physical_sum", "equal_sum", "max_metric")
BLOCK_NAMES = ("egi29", "egi57", "fre")


@dataclass(slots=True)
class OfflineState:
    state_id: str
    source: str
    case_name: str
    seed: int | None
    basis_count: int | None
    accepted: bool | None
    final: bool
    yielded_rmse_mpa: float
    high_plastic_rmse_mpa: float
    roi_rmse_mpa: float
    scores: dict[str, float]


def main() -> None:
    args = _parse_args()
    campaign_root = args.campaign_root.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        (campaign_root / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    prepared = dataset / "prepared"
    experiment = ExperimentData.load_from_file(prepared / "experiment_data.yaml")
    known_raw = load_known_parameter_maps(prepared / "known_parameter_maps.npz")
    if known_raw is None:
        raise RuntimeError("Known parameter maps are required.")
    known = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in known_raw.items()
    }
    mask = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(
        experiment.specimen_geometry.x,
        experiment.specimen_geometry.y,
    )
    run_items = _completed_runs(campaign_root, manifest)
    if not run_items:
        raise RuntimeError("No completed campaign runs were found.")
    reference = load_identification_result(run_items[0][1])
    law = load_constitutive_law_from_result(reference)
    metrics = _create_metrics(experiment)
    baselines = _extract_baselines(reference)
    yielded, high_plastic = _active_masks(experiment, law, known, mask)
    truth_metrics = _load_truth_metrics(args.analysis / "state_metrics.csv")

    phase_zero = reference.history.phases[0].solve_results[-1]
    if phase_zero.final_snapshot is None:
        raise RuntimeError("The phase-0 reference snapshot is unavailable.")
    baseline_maps = _complete_maps(
        evaluate_snapshot_parameter_maps(phase_zero.final_snapshot, experiment),
        known,
    )
    baseline_features = _raw_features(
        law, baseline_maps, experiment, metrics, baselines
    )

    states: list[OfflineState] = []
    total = sum(
        len(load_identification_result(path).history.phases[-1].solve_results)
        for _, path in run_items
    ) + 31
    completed = 0
    for metadata, result_path in run_items:
        result = load_identification_result(result_path)
        solve_states = []
        for solve in result.history.phases[-1].solve_results:
            if solve.final_snapshot is None:
                continue
            state_id = f"{metadata['name']}/solve_{solve.solve_iteration}"
            maps = _complete_maps(
                evaluate_snapshot_parameter_maps(solve.final_snapshot, experiment),
                known,
            )
            features = _raw_features(law, maps, experiment, metrics, baselines)
            solve_states.append(
                _state(
                    state_id,
                    "campaign",
                    str(metadata["name"]),
                    int(metadata["seed"]),
                    _basis_count(solve.final_snapshot),
                    bool(solve.accepted),
                    False,
                    features,
                    baseline_features,
                    truth_metrics,
                )
            )
            completed += 1
            _progress(completed, total, state_id)
        accepted = [state for state in solve_states if state.accepted]
        if accepted:
            accepted[-1].final = True
        states.extend(solve_states)

    for label, maps in _controlled_states(known, experiment, mask):
        state_id = f"controlled/{label}"
        features = _raw_features(law, maps, experiment, metrics, baselines)
        states.append(
            _state(
                state_id,
                "controlled",
                label,
                None,
                None,
                None,
                False,
                features,
                baseline_features,
                truth_metrics,
            )
        )
        completed += 1
        _progress(completed, total, state_id)

    rankings = _rankings(states)
    selections = _selections(states)
    _write_scores(output / "offline_state_scores.csv", states)
    _write_rankings(output / "candidate_rankings.csv", rankings)
    _write_selections(output / "trajectory_selections.csv", selections)
    summary = {
        "states": len(states),
        "candidate_scores": len(states[0].scores),
        "rankings": rankings,
        "selections": selections,
    }
    (output / "offline_selector_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _plots(output, rankings, selections)
    print(json.dumps({
        "output": str(output),
        "states": len(states),
        "candidate_scores": len(states[0].scores),
        "best_campaign_yielded": rankings[0]["candidate"],
    }, indent=2))


def _load_truth_metrics(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["state_id"]: row for row in csv.DictReader(stream)}


def _raw_features(law, maps, experiment, metrics, baselines):
    stress = law.calculate_stress(experiment.strain, maps)
    force_metric, egi_metrics = metrics
    egi_fields = []
    egi_time_weights = []
    for metric in egi_metrics:
        result = metric.evaluate_equilibrium_gap(stress)
        egi_fields.append(np.asarray(result.normalised_gap, dtype=np.float64))
        egi_time_weights.append(np.asarray(
            result.metric_result.additional_fields["temporal_weights"],
            dtype=np.float64,
        ))
    force_result = force_metric.evaluate_force_recon_error(stress, experiment)
    force_meta = force_result.metric_result.additional_fields
    force_field = np.asarray(force_meta["normalised_residual"], dtype=np.float64)
    force_time = np.asarray(force_meta["temporal_weights"], dtype=np.float64)
    force_space = np.asarray(force_meta["spatial_weights"], dtype=np.float64)

    plasticity = compute_plasticity_diagnostics(experiment, law, maps)
    if plasticity is None:
        raise RuntimeError("Plasticity diagnostics are unavailable.")
    plastic = np.asarray(plasticity.equivalent_plastic_strain, dtype=np.float64)

    fields = [*egi_fields, force_field]
    time_weights = [*egi_time_weights, force_time]
    space_weights = [None, None, force_space]
    features = {name: [] for name in AGGREGATIONS}
    per_time_rms = []
    for index, (field, temporal, spatial) in enumerate(
        zip(fields, time_weights, space_weights, strict=True)
    ):
        features["rms"].append(_weighted_rms(field, temporal, spatial))
        features["hierarchical_mean"].append(
            _hierarchical_mean(field, temporal, spatial)
        )
        features["time_p90"].append(_time_percentile(field, temporal, spatial, 0.90))
        features["tail_p90"].append(_tail_percentile(field, temporal, spatial, 0.90))
        features["tail_p95"].append(_tail_percentile(field, temporal, spatial, 0.95))
        features["coherent_rms"].append(
            _coherent_rms(field, temporal, spatial)
        )
        features["plastic_active_rms"].append(
            _plastic_active_rms(field, temporal, spatial, plastic, index == 2)
        )
        per_time_rms.append(_spatial_rms(field, spatial))

    _, current_scalars = _residual_blocks(
        law, maps, experiment, metrics, baselines
    )
    return {
        "current_objective": np.asarray(current_scalars, dtype=np.float64),
        **{
            name: np.asarray(values, dtype=np.float64)
            for name, values in features.items()
        },
        "per_time_rms": np.asarray(per_time_rms, dtype=np.float64),
        "load_weights": _normalise(force_time),
    }


def _normalise(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    total = float(np.nansum(values))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(values.shape, 1.0 / values.size)
    return values / total


def _observation_weights(field, temporal, spatial):
    shape = np.asarray(field).shape
    time = _normalise(temporal)
    if len(shape) == 3:
        weights = np.broadcast_to(time[:, None, None], shape).copy()
    else:
        space = _normalise(spatial) if spatial is not None else np.full(shape[1], 1.0 / shape[1])
        weights = time[:, None] * space[None, :]
    valid = np.isfinite(field)
    weights[~valid] = 0.0
    return _normalise(weights)


def _weighted_rms(field, temporal, spatial):
    weights = _observation_weights(field, temporal, spatial)
    return float(np.sqrt(np.nansum(weights * np.asarray(field) ** 2)))


def _spatial_rms(field, spatial):
    values = np.asarray(field, dtype=np.float64)
    if values.ndim == 3:
        return np.sqrt(np.nanmean(values**2, axis=(1, 2)))
    weights = _normalise(spatial) if spatial is not None else np.full(values.shape[1], 1.0 / values.shape[1])
    return np.sqrt(np.nansum(values**2 * weights[None, :], axis=1))


def _hierarchical_mean(field, temporal, spatial):
    per_time = _spatial_rms(field, spatial)
    return float(np.nansum(_normalise(temporal) * per_time))


def _weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    cumulative = np.cumsum(weights[order])
    target = quantile * cumulative[-1]
    return float(values[min(np.searchsorted(cumulative, target), values.size - 1)])


def _time_percentile(field, temporal, spatial, quantile):
    return _weighted_quantile(_spatial_rms(field, spatial), temporal, quantile)


def _tail_percentile(field, temporal, spatial, quantile):
    weights = _observation_weights(field, temporal, spatial)
    return _weighted_quantile(np.abs(field), weights, quantile)


def _coherent_rms(field, temporal, spatial):
    values = np.asarray(field, dtype=np.float64)
    filled = np.nan_to_num(values, nan=0.0)
    if values.ndim == 3:
        coherent = gaussian_filter(filled, sigma=(0.0, 2.0, 2.0), mode="nearest")
        valid_fraction = gaussian_filter(
            np.isfinite(values).astype(float), sigma=(0.0, 2.0, 2.0), mode="nearest"
        )
    else:
        coherent = gaussian_filter1d(filled, sigma=2.0, axis=1, mode="nearest")
        valid_fraction = gaussian_filter1d(
            np.isfinite(values).astype(float), sigma=2.0, axis=1, mode="nearest"
        )
    coherent = np.divide(
        coherent,
        valid_fraction,
        out=np.full_like(coherent, np.nan),
        where=valid_fraction > 0.25,
    )
    return _weighted_rms(coherent, temporal, spatial)


def _plastic_active_rms(field, temporal, spatial, plastic, is_force):
    values = np.asarray(field, dtype=np.float64)
    activity = np.maximum(np.asarray(plastic, dtype=np.float64), 0.0)
    positive = activity[np.isfinite(activity) & (activity > 0.0)]
    scale = float(np.percentile(positive, 90.0)) if positive.size else 1.0
    activity = np.clip(activity / max(scale, np.finfo(float).eps), 0.0, 1.0)
    if is_force:
        active_time = np.nanmean(activity, axis=(1, 2))
        active_time /= max(float(np.nanmax(active_time)), np.finfo(float).eps)
        adjusted_time = _normalise(temporal) * (0.05 + 0.95 * active_time)
        return _weighted_rms(values, adjusted_time, spatial)
    weights = _observation_weights(values, temporal, spatial)
    weights *= 0.05 + 0.95 * activity
    weights = _normalise(weights)
    return float(np.sqrt(np.nansum(weights * values**2)))


def _scores(features, baseline):
    scores = {"current": _objective_from_scalars(features["current_objective"])}
    for aggregation in AGGREGATIONS:
        normalised = np.divide(
            features[aggregation],
            baseline[aggregation],
            out=np.full(3, np.nan),
            where=np.asarray(baseline[aggregation]) > 0.0,
        )
        scores[f"{aggregation}__physical_sum"] = float(
            np.dot(COMBINATION_WEIGHTS, normalised)
        )
        scores[f"{aggregation}__equal_sum"] = float(np.mean(normalised))
        scores[f"{aggregation}__max_metric"] = float(np.max(normalised))
        for index, block_name in enumerate(BLOCK_NAMES):
            scores[f"{aggregation}__{block_name}"] = float(normalised[index])
    per_time = np.divide(
        features["per_time_rms"],
        baseline["per_time_rms"],
        out=np.full_like(features["per_time_rms"], np.nan),
        where=np.asarray(baseline["per_time_rms"]) > 0.0,
    )
    combined_time = np.dot(COMBINATION_WEIGHTS, per_time)
    for index, value in enumerate(combined_time):
        scores[f"load_step_{index:02d}"] = float(value)
    load_weights = np.asarray(features["load_weights"], dtype=np.float64)
    high_load = load_weights >= np.median(load_weights)
    scores["high_load_mean"] = float(np.average(
        combined_time[high_load], weights=load_weights[high_load]
    ))
    scores["load_step_max"] = float(np.max(combined_time))
    scores["load_step_p90"] = float(np.percentile(combined_time, 90.0))
    return scores


def _state(
    state_id, source, case_name, seed, basis_count, accepted, final,
    features, baseline, truth_metrics,
):
    truth = truth_metrics[state_id]
    return OfflineState(
        state_id=state_id,
        source=source,
        case_name=case_name,
        seed=seed,
        basis_count=basis_count,
        accepted=accepted,
        final=final,
        yielded_rmse_mpa=float(truth["yielded_rmse_mpa"]),
        high_plastic_rmse_mpa=float(truth["high_plastic_rmse_mpa"]),
        roi_rmse_mpa=float(truth["roi_rmse_mpa"]),
        scores=_scores(features, baseline),
    )


def _progress(completed, total, state_id):
    if completed == 1 or completed % 10 == 0 or completed == total:
        print(f"[{completed:3d}/{total}] {state_id}", flush=True)


def _pairwise_accuracy(scores, errors):
    correct = 0
    compared = 0
    for left in range(len(scores)):
        for right in range(left + 1, len(scores)):
            score_delta = scores[left] - scores[right]
            error_delta = errors[left] - errors[right]
            if score_delta == 0.0 or error_delta == 0.0:
                continue
            compared += 1
            correct += (score_delta > 0.0) == (error_delta > 0.0)
    return correct / compared if compared else float("nan")


def _adjacent_accuracy(states, candidate, error_name, minimum_after_basis=0):
    cases = {}
    for state in states:
        if state.source == "campaign" and state.accepted:
            cases.setdefault(state.case_name, []).append(state)
    correct = 0
    compared = 0
    for rows in cases.values():
        ordered = sorted(rows, key=lambda row: row.basis_count or -1)
        for before, after in zip(ordered, ordered[1:]):
            if (after.basis_count or -1) < minimum_after_basis:
                continue
            score_delta = after.scores[candidate] - before.scores[candidate]
            error_delta = getattr(after, error_name) - getattr(before, error_name)
            if score_delta == 0.0 or error_delta == 0.0:
                continue
            compared += 1
            correct += (score_delta > 0.0) == (error_delta > 0.0)
    return correct / compared if compared else float("nan")


def _rankings(states):
    candidates = list(states[0].scores)
    output = []
    subsets = {
        "campaign": [s for s in states if s.source == "campaign"],
        "campaign_late": [
            s for s in states
            if s.source == "campaign" and (s.basis_count or -1) >= 5
        ],
        "controlled": [s for s in states if s.source == "controlled"],
        "final": [s for s in states if s.final],
    }
    for candidate in candidates:
        row = {"candidate": candidate}
        for label, selected in subsets.items():
            scores = [state.scores[candidate] for state in selected]
            for error_name in ("yielded_rmse_mpa", "high_plastic_rmse_mpa"):
                errors = [getattr(state, error_name) for state in selected]
                coefficient, p_value = spearmanr(scores, errors)
                prefix = f"{label}_{error_name.removesuffix('_mpa')}"
                row[f"{prefix}_spearman"] = float(coefficient)
                row[f"{prefix}_p"] = float(p_value)
                row[f"{prefix}_pairwise"] = _pairwise_accuracy(scores, errors)
        row["adjacent_yielded_pairwise"] = _adjacent_accuracy(
            states, candidate, "yielded_rmse_mpa"
        )
        row["adjacent_high_plastic_pairwise"] = _adjacent_accuracy(
            states, candidate, "high_plastic_rmse_mpa"
        )
        row["late_adjacent_yielded_pairwise"] = _adjacent_accuracy(
            states, candidate, "yielded_rmse_mpa", minimum_after_basis=7
        )
        row["late_adjacent_high_plastic_pairwise"] = _adjacent_accuracy(
            states, candidate, "high_plastic_rmse_mpa", minimum_after_basis=7
        )
        output.append(row)
    output.sort(
        key=lambda row: (
            -row["campaign_yielded_rmse_spearman"],
            -row["campaign_late_yielded_rmse_spearman"],
        )
    )
    return output


def _selections(states):
    cases = {}
    for state in states:
        if state.source == "campaign" and state.accepted and state.case_name.startswith("spd_sensitivity_gate0p0pct"):
            cases.setdefault(state.case_name, []).append(state)
    output = []
    for candidate in states[0].scores:
        selected = [min(rows, key=lambda state: state.scores[candidate]) for rows in cases.values()]
        output.append({
            "candidate": candidate,
            "median_basis_count": float(median(state.basis_count for state in selected if state.basis_count is not None)),
            "basis_counts": ",".join(str(state.basis_count) for state in selected),
            "median_yielded_rmse_mpa": float(median(state.yielded_rmse_mpa for state in selected)),
            "median_high_plastic_rmse_mpa": float(median(state.high_plastic_rmse_mpa for state in selected)),
            "median_roi_rmse_mpa": float(median(state.roi_rmse_mpa for state in selected)),
        })
    output.sort(key=lambda row: (row["median_yielded_rmse_mpa"], row["median_high_plastic_rmse_mpa"]))
    return output


def _write_scores(path, states):
    candidates = list(states[0].scores)
    fields = [
        "state_id", "source", "case_name", "seed", "basis_count", "accepted",
        "final", "yielded_rmse_mpa", "high_plastic_rmse_mpa", "roi_rmse_mpa",
        *candidates,
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for state in states:
            row = {key: getattr(state, key) for key in fields if hasattr(state, key)}
            row.update(state.scores)
            writer.writerow(row)


def _write_rankings(path, rankings):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rankings[0]))
        writer.writeheader()
        writer.writerows(rankings)


def _write_selections(path, selections):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selections[0]))
        writer.writeheader()
        writer.writerows(selections)


def _plots(output, rankings, selections):
    top = rankings[:10]
    figure, axis = plt.subplots(figsize=(10, 5), layout="constrained")
    positions = np.arange(len(top))
    axis.barh(
        positions,
        [row["campaign_yielded_rmse_spearman"] for row in top],
        color="tab:blue",
    )
    axis.set_yticks(positions, [row["candidate"] for row in top])
    axis.invert_yaxis()
    axis.set(xlabel="Spearman correlation with yielded RMSE", title="Top offline scalar reductions")
    axis.grid(axis="x", alpha=0.25)
    figure.savefig(output / "candidate_ranking.png", dpi=180)
    plt.close(figure)

    selected = sorted(selections, key=lambda row: row["median_yielded_rmse_mpa"])[:10]
    figure, axis = plt.subplots(figsize=(10, 5), layout="constrained")
    axis.barh(
        np.arange(len(selected)),
        [row["median_yielded_rmse_mpa"] for row in selected],
        color="tab:green",
    )
    axis.set_yticks(np.arange(len(selected)), [row["candidate"] for row in selected])
    axis.invert_yaxis()
    axis.set(xlabel="Median yielded RMSE [MPa]", title="Minimum-score trajectory selection")
    axis.grid(axis="x", alpha=0.25)
    figure.savefig(output / "trajectory_selection.png", dpi=180)
    plt.close(figure)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
