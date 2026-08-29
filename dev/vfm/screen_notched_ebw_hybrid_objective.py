"""Screen noise-aware hybrid EGI/FRE objectives for synthetic notched EBW."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

from pyvale.vfm import (
    EquilibriumGapMetric, ExperimentData, SliceConfig,
    SliceWiseForceReconstructionMetric, load_identification_result,
)
from pyvale.vfm.campaignprogress import ProgressEstimate, atomic_write_json
from pyvale.vfm.loadregimes import (
    LoadRegimeThresholds,
    resolve_load_regimes,
    resolve_relative_load_regimes,
)
from pyvale.vfm.postprocessing import (
    compute_plasticity_diagnostics, evaluate_snapshot_parameter_maps,
    load_constitutive_law_from_result, load_known_parameter_maps,
)
from pyvale.vfm.residualfeatures import coherent_rms, weighted_cvar_abs, weighted_rms

from analyse_notched_ebw_component_library import (
    _independent_states, _optimiser_late_states, _property_errors,
)
from analyse_notched_ebw_gate_campaign import _active_masks, _complete_maps


@dataclass(slots=True)
class FeatureRow:
    name: str
    source: str
    split: str
    family: str
    basis_count: int | None
    seed: int | None
    roi_rmse_mpa: float
    yielded_rmse_mpa: float
    high_plastic_rmse_mpa: float
    hardening_rmse_mpa: float
    features: dict[str, float]


def main() -> None:
    args = _parse_args()
    dataset = args.dataset.expanduser().resolve()
    campaign = args.campaign_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared = dataset / "prepared"
    experiment = ExperimentData.load_from_file(prepared / "experiment_data.yaml")
    known_raw = load_known_parameter_maps(prepared / "known_parameter_maps.npz")
    if known_raw is None:
        raise RuntimeError("Known parameter maps are required for the synthetic screen.")
    known = {name: np.asarray(value, dtype=np.float64) for name, value in known_raw.items()}
    result_path = next(iter(sorted(campaign.glob("*/identification_result.yaml"))))
    reference = load_identification_result(result_path)
    law = load_constitutive_law_from_result(reference)
    mask = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(
        experiment.specimen_geometry.x, experiment.specimen_geometry.y
    )
    yielded, high_plastic = _active_masks(experiment, law, known, mask)
    windows = _resolve_windows(experiment, args.window_lengths_mm)
    metrics = _create_metrics(experiment, windows, args.force_slices)
    phase_zero_maps = _phase_zero_maps(reference, experiment, known)
    regimes = _phase_zero_regimes(experiment, law, phase_zero_maps, args)

    cache_configuration = {
        "schema": "notched_ebw_hybrid_features_v2",
        "dataset": str(dataset),
        "campaign_root": str(campaign),
        "windows": windows,
        "force_slices": args.force_slices,
        "load_regimes": regimes.diagnostics(),
    }
    _validate_or_write_cache_configuration(
        output / "screen_configuration.json",
        cache_configuration,
        resume=args.resume,
        cached_artifacts=(
            output / "state_feature_rows.csv",
            output / "noise_floors.json",
        ),
    )

    states = [
        *_independent_states(experiment, law, known, mask),
        *_optimiser_late_states(campaign, experiment, known),
    ]
    if args.max_states:
        states = states[:args.max_states]
    feature_path = output / "state_feature_rows.csv"
    feature_rows = _read_feature_rows(feature_path) if args.resume and feature_path.is_file() else []
    completed_states = {row.name for row in feature_rows}
    total = len(states) + args.noise_replicates
    started = time.monotonic()
    for index, state in enumerate(states, start=1):
        if state.name in completed_states:
            continue
        errors = _property_errors(state.maps, known, mask, yielded, high_plastic)
        feature_rows.append(FeatureRow(
            name=state.name, source=state.source, split=state.split,
            family=state.family, basis_count=state.basis_count, seed=state.seed,
            **errors,
            features=_features(law, state.maps, experiment, metrics, windows, regimes),
        ))
        _write_feature_rows(feature_path, feature_rows)
        if index == 1 or index % 5 == 0 or index == len(states):
            print(ProgressEstimate.from_counts(index, total, started).line(prefix="screen") + f" state={state.name}", flush=True)

    phase_zero_features = _features(
        law, phase_zero_maps, experiment, metrics, windows, regimes
    )
    floors_path = output / "noise_floors.json"
    if args.resume and floors_path.is_file():
        noise_floors = json.loads(floors_path.read_text(encoding="utf-8"))
    else:
        noise_floors = _noise_floors(
            experiment, law, known, windows, regimes, args, started, len(states), total
        )
        floors_path.write_text(json.dumps(noise_floors, indent=2) + "\n", encoding="utf-8")
    candidates, scores = _candidate_scores(
        feature_rows, phase_zero_features, noise_floors, windows, args
    )
    _write_csv(output / "candidate_objective_scores.csv", scores)
    window_scores = _window_information(scores)
    _write_csv(output / "window_information_scores.csv", window_scores)
    selected = _select_candidates(candidates, scores, windows, regimes, noise_floors, args)
    (output / "selected_windows.json").write_text(
        json.dumps(selected["windows"], indent=2) + "\n", encoding="utf-8"
    )
    (output / "selected_objectives.json").write_text(
        json.dumps(selected["objectives"], indent=2) + "\n", encoding="utf-8"
    )
    for key, filename in (
        ("raw_parsimonious", "raw_parsimonious_objective.json"),
        ("raw_information_rich", "raw_information_rich_objective.json"),
    ):
        (output / filename).write_text(
            json.dumps(selected["objectives"][key], indent=2) + "\n",
            encoding="utf-8",
        )
    atomic_write_json(output / "screen_manifest.json", {
        "tool": "screen_notched_ebw_hybrid_objective", "status": "complete",
        "dataset": str(dataset), "campaign_root": str(campaign),
        "window_lengths_mm": list(args.window_lengths_mm),
        "window_pixels": windows, "load_regimes": regimes.diagnostics(),
        "noise_replicates": args.noise_replicates,
        "states": len(feature_rows), "candidates": len(candidates),
    })
    print(f"screen complete output={output} candidates={len(candidates)}", flush=True)


def _validate_or_write_cache_configuration(
    path: Path,
    configuration: dict,
    *,
    resume: bool,
    cached_artifacts: tuple[Path, ...],
) -> None:
    """Prevent frame-dependent features being reused after regime changes."""

    if resume and any(item.is_file() for item in cached_artifacts):
        if not path.is_file():
            raise RuntimeError(
                "Refusing to resume legacy screen cache without "
                f"{path.name}; use a new output directory."
            )
        previous = json.loads(path.read_text(encoding="utf-8"))
        # ``atomic_write_json`` adds operational provenance which is not part
        # of the scientific cache key.
        previous.pop("updated_at", None)
        if previous != configuration:
            raise RuntimeError(
                "Screen cache configuration differs from this run; use a new "
                "output directory rather than mixing regime-dependent features."
            )
    atomic_write_json(path, configuration)


def _resolve_windows(experiment, lengths):
    x = np.asarray(experiment.specimen_geometry.x, dtype=np.float64)
    y = np.asarray(experiment.specimen_geometry.y, dtype=np.float64)
    dx = float(np.nanmedian(np.abs(np.diff(x, axis=1))))
    dy = float(np.nanmedian(np.abs(np.diff(y, axis=0))))
    from pyvale.vfm.residualfeatures import physical_length_to_odd_pixels
    return {
        f"{length:g}mm": [
            physical_length_to_odd_pixels(length, dy),
            physical_length_to_odd_pixels(length, dx),
        ] for length in lengths
    }


def _create_metrics(experiment, windows, force_slices):
    force = SliceWiseForceReconstructionMetric(
        slice_config=SliceConfig(axis="x", num_slices=force_slices)
    )
    egi = [EquilibriumGapMetric(window_size=tuple(size)) for size in windows.values()]
    for metric in [force, *egi]:
        metric.initialise(experiment)
    return force, egi


def _phase_zero_maps(result, experiment, known):
    solve = result.history.phases[0].solve_results[-1]
    if solve.final_snapshot is None:
        raise RuntimeError("Phase-0 snapshot is unavailable.")
    return _complete_maps(evaluate_snapshot_parameter_maps(solve.final_snapshot, experiment), known)


def _phase_zero_regimes(experiment, law, maps, args):
    plasticity = compute_plasticity_diagnostics(experiment, law, maps)
    if plasticity is None:
        raise RuntimeError("Phase-0 plasticity diagnostics are unavailable.")
    plastic = np.asarray(plasticity.equivalent_plastic_strain, dtype=np.float64)
    finite = np.isfinite(plastic)
    yielded = finite & (plastic > args.plastic_strain_threshold)
    fraction = np.sum(yielded, axis=(1, 2)) / np.maximum(np.sum(finite, axis=(1, 2)), 1)
    thresholds = LoadRegimeThresholds(
        args.onset_fraction,
        args.developed_fraction,
        args.late_fraction,
    )
    if args.regime_mode == "relative":
        return resolve_relative_load_regimes(
            fraction,
            thresholds,
            minimum_frames=args.minimum_regime_frames,
        )
    return resolve_load_regimes(fraction, thresholds)


def _features(law, maps, experiment, metrics, windows, regimes):
    stress = law.calculate_stress(experiment.strain, maps)
    force_metric, egi_metrics = metrics
    output = {}
    for label, metric in zip(windows, egi_metrics, strict=True):
        result = metric.evaluate_equilibrium_gap(stress)
        metadata = result.metric_result.additional_fields
        _block_features(output, f"egi_{label}", result.normalised_gap, metadata, regimes)
    force_result = force_metric.evaluate_force_recon_error(stress, experiment).metric_result
    metadata = force_result.additional_fields
    _block_features(output, "fre", metadata["normalised_residual"], metadata, regimes)
    return output


def _block_features(output, prefix, field, metadata, regimes):
    values = np.asarray(field, dtype=np.float64)
    temporal = np.asarray(metadata.get("temporal_weights", np.ones(values.shape[0])), dtype=np.float64)
    spatial = metadata.get("spatial_weights")
    for regime_name in ("pre_yield", "onset", "developed", "late"):
        indices = np.asarray(regimes.indices(regime_name), dtype=np.int64)
        block = values[indices]
        weights = _weights(block, temporal[indices], spatial)
        key = f"{prefix}__{regime_name}"
        output[f"{key}__rms"] = weighted_rms(block, weights).value
        output[f"{key}__cvar90"] = weighted_cvar_abs(block, weights, quantile=0.90).value
        output[f"{key}__cvar95"] = weighted_cvar_abs(block, weights, quantile=0.95).value
        axes = (-2, -1) if block.ndim == 3 else (-1,)
        output[f"{key}__coherent_rms"] = coherent_rms(
            block, weights, sigma_pixels=tuple(2.0 for _ in axes), spatial_axes=axes
        ).value


def _weights(values, temporal, spatial):
    weights = np.broadcast_to(
        np.asarray(temporal, dtype=np.float64).reshape((-1,) + (1,) * (values.ndim - 1)),
        values.shape,
    ).copy()
    if spatial is not None:
        spatial = np.asarray(spatial, dtype=np.float64)
        weights *= np.broadcast_to(spatial.reshape((1,) * (values.ndim - spatial.ndim) + spatial.shape), values.shape)
    return weights


def _noise_floors(experiment, law, known, windows, regimes, args, started, offset, total):
    if args.noise_replicates == 0:
        return {key: 0.0 for key in _features(law, known, experiment, _create_metrics(experiment, windows, args.force_slices), windows, regimes)}
    model = json.loads(args.noise_model.read_text(encoding="utf-8"))
    rows = []
    for replicate in range(args.noise_replicates):
        noisy = copy.deepcopy(experiment)
        _apply_noise(noisy, model, args.noise_scale, args.random_seed + replicate)
        metrics = _create_metrics(noisy, windows, args.force_slices)
        rows.append(_features(law, known, noisy, metrics, windows, regimes))
        print(ProgressEstimate.from_counts(offset + replicate + 1, total, started).line(prefix="screen") + f" noise={replicate + 1}/{args.noise_replicates}", flush=True)
    return {key: float(np.median([row[key] for row in rows])) for key in rows[0]}


def _apply_noise(experiment, model, scale, seed):
    rng = np.random.default_rng(seed)
    strain = np.asarray(experiment.strain, dtype=np.float64).copy()
    valid = np.all(np.isfinite(strain), axis=(0, 1))
    x = experiment.specimen_geometry.x; y = experiment.specimen_geometry.y
    dx = float(np.nanmedian(np.abs(np.diff(x, axis=1)))); dy = float(np.nanmedian(np.abs(np.diff(y, axis=0))))
    for component, name in enumerate(("exx", "eyy", "exy")):
        config = model["components"][name]
        sigma = float(config["sigma"]) * scale
        smooth = config["gaussian_filter_sigma_mm"]
        for timestep in range(strain.shape[0]):
            sample = gaussian_filter(rng.standard_normal(strain.shape[2:]), sigma=(float(smooth["y"])/dy, float(smooth["x"])/dx), mode="reflect")
            sample -= np.mean(sample[valid]); sample /= max(np.std(sample[valid]), np.finfo(float).eps)
            strain[timestep, component, valid] += sigma * sample[valid]
    experiment.strain = strain
    force = np.asarray(experiment.boundary_conditions.force, dtype=np.float64).copy()
    noise = rng.normal(0.0, float(model["force"]["sigma_n"]) * scale, force.shape[0])
    if force.ndim == 1: force += noise
    else: force[:, 0] += noise
    experiment.boundary_conditions.force = force


def _candidate_scores(rows, reference, floors, windows, args):
    labels = list(windows)
    sets = [(labels[0], labels[-1]), (labels[0], labels[1], labels[-1]), (labels[0], labels[2], labels[-1])]
    candidates = []
    scores = []
    for supports in sets:
        middle = supports[1] if len(supports) == 3 else supports[0]
        terms = [
            f"egi_{supports[0]}__onset__cvar90",
            f"egi_{middle}__late__rms",
            f"egi_{supports[-1]}__onset__cvar90",
            "fre__late__coherent_rms",
        ]
        global_terms = [*[f"egi_{label}__developed__rms" for label in supports], "fre__developed__rms"]
        for alpha in args.alphas:
            name = f"raw_{'_'.join(supports)}_a{alpha:g}"
            candidates.append({"name": name, "supports": supports, "terms": terms, "global_terms": global_terms, "alpha": alpha})
            for row in rows:
                material = [_normalised(row.features[key], floors[key], reference[key]) for key in terms]
                global_values = [_normalised(row.features[key], floors[key], reference[key]) for key in global_terms]
                material_cost = 0.9 * max(material) + 0.1 * float(np.mean(material))
                score = (1.0 - alpha) * float(np.mean(global_values)) + alpha * material_cost
                scores.append({
                    "candidate": name, "state": row.name, "source": row.source,
                    "split": row.split, "family": row.family,
                    "basis_count": row.basis_count, "seed": row.seed,
                    "score": score, "global_cost": float(np.mean(global_values)),
                    "material_cost": material_cost,
                    "yielded_rmse_mpa": row.yielded_rmse_mpa,
                    "high_plastic_rmse_mpa": row.high_plastic_rmse_mpa,
                })
    return candidates, scores


def _normalised(value, floor, reference):
    return max(float(value) - float(floor), 0.0) / max(float(reference) - float(floor), 1e-12)


def _window_information(scores):
    output = []
    for candidate in sorted({row["candidate"] for row in scores}):
        selected = [row for row in scores if row["candidate"] == candidate]
        groups = {
            "development": [row for row in selected if row["split"] == "development"],
            "validation": [row for row in selected if row["split"] == "validation"],
        }
        diagnostic_bfs = sorted({row["basis_count"] for row in selected if row["split"] == "diagnostic" and row["basis_count"] is not None})
        groups.update({
            f"diagnostic_bf{bf}": [row for row in selected if row["split"] == "diagnostic" and row["basis_count"] == bf]
            for bf in diagnostic_bfs
        })
        for subset, rows in groups.items():
            if len(rows) < 3: continue
            for target in ("yielded_rmse_mpa", "high_plastic_rmse_mpa"):
                rho = spearmanr([row["score"] for row in rows], [row[target] for row in rows]).statistic
                output.append({"candidate": candidate, "subset": subset, "target": target, "states": len(rows), "spearman_r": float(rho), "pairwise_accuracy": _pairwise(rows, target)})
    return output


def _pairwise(rows, target):
    scores = np.asarray([row["score"] for row in rows]); errors = np.asarray([row[target] for row in rows])
    ds = scores[:, None] - scores[None, :]; de = errors[:, None] - errors[None, :]
    valid = np.triu(np.ones(ds.shape, dtype=bool), 1) & (ds != 0) & (de != 0)
    return float(np.mean(np.sign(ds[valid]) == np.sign(de[valid]))) if np.any(valid) else float("nan")


def _select_candidates(candidates, scores, windows, regimes, floors, args):
    information = _window_information(scores)
    def merit(candidate):
        rows = [row for row in information if row["candidate"] == candidate["name"] and row["subset"] in {"development", "validation"}]
        return float(np.nanmean([row["spearman_r"] for row in rows]))
    ordered = sorted(candidates, key=merit, reverse=True)
    raw = ordered[0]
    objective = _objective_payload(raw, windows, regimes, floors)
    parsimonious = max(
        (candidate for candidate in candidates if len(candidate["supports"]) == 2),
        key=merit,
    )
    information_rich = max(
        (
            candidate for candidate in candidates
            if len(candidate["supports"]) == 3
            and "5.8mm" in candidate["supports"]
        ),
        key=merit,
    )
    projected = copy.deepcopy(objective)
    projected["name"] = raw["name"] + "_projected_candidate"
    projected["projection"] = {"status": "requires online native-DOF preparation", "reduction": "yield_unique_projected_rms", "relative_tolerance": 1e-8}
    return {
        "windows": {"selected_labels": list(raw["supports"]), "all_candidates": windows, "selection_merit": merit(raw)},
        "objectives": {
            "raw": objective,
            "raw_parsimonious": _objective_payload(
                parsimonious, windows, regimes, floors
            ),
            "raw_information_rich": _objective_payload(
                information_rich, windows, regimes, floors
            ),
            "projected": projected,
            "ranking": [
                {"name": item["name"], "merit": merit(item)}
                for item in ordered
            ],
        },
    }


def _objective_payload(candidate, windows, regimes, floors):
    supports = candidate["supports"]; middle = supports[1] if len(supports) == 3 else supports[0]
    feature_specs = [
        ("fine_egi_onset_cvar90", supports[0], "cvar_abs", regimes.onset, 0.90),
        ("middle_egi_late_rms", middle, "rms", regimes.late, None),
        ("broad_egi_onset_cvar90", supports[-1], "cvar_abs", regimes.onset, 0.90),
        ("fre_late_coherent", None, "coherent_rms", regimes.late, None),
    ]
    features = []
    for name, support, reduction, frames, quantile in feature_specs:
        result_index = 0 if support is None else 1 + list(supports).index(support)
        floor_key = "fre__late__coherent_rms" if support is None else f"egi_{support}__{'late__rms' if 'late_rms' in name else 'onset__cvar90'}"
        item = {"name": name, "metric_result_index": result_index, "reduction": reduction, "frame_indices": list(frames), "weight": 1.0, "noise_floor": floors[floor_key]}
        if quantile is not None: item["quantile"] = quantile
        if reduction == "coherent_rms": item.update({"sigma_pixels": 2.0, "spatial_axes": [-1]})
        features.append(item)
    return {"name": candidate["name"], "alpha": candidate["alpha"], "smooth_max_temperature": 0.1, "mean_fraction": 0.1, "positive_part_temperature": 0.001, "egi_windows": [windows[label] for label in supports], "features": features}


def _write_feature_rows(path, rows):
    flat = [{**{key: value for key, value in asdict(row).items() if key != "features"}, **row.features} for row in rows]
    _write_csv(path, flat)


def _read_feature_rows(path):
    metadata = {"name", "source", "split", "family", "basis_count", "seed", "roi_rmse_mpa", "yielded_rmse_mpa", "high_plastic_rmse_mpa", "hardening_rmse_mpa"}
    output = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            output.append(FeatureRow(
                name=row["name"], source=row["source"], split=row["split"], family=row["family"],
                basis_count=None if not row["basis_count"] else int(row["basis_count"]),
                seed=None if not row["seed"] else int(row["seed"]),
                roi_rmse_mpa=float(row["roi_rmse_mpa"]), yielded_rmse_mpa=float(row["yielded_rmse_mpa"]),
                high_plastic_rmse_mpa=float(row["high_plastic_rmse_mpa"]), hardening_rmse_mpa=float(row["hardening_rmse_mpa"]),
                features={key: float(value) for key, value in row.items() if key not in metadata},
            ))
    return output


def _write_csv(path, rows):
    if not rows: return
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--noise-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-lengths-mm", type=lambda value: tuple(float(item) for item in value.split(",")), default=(1.4, 3.0, 5.8, 11.4))
    parser.add_argument("--alphas", type=lambda value: tuple(float(item) for item in value.split(",")), default=(0.25, 0.5, 0.75))
    parser.add_argument("--noise-replicates", type=int, default=8)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=20260829)
    parser.add_argument("--force-slices", type=int, default=63)
    parser.add_argument("--max-states", type=int, default=0, help="Smoke-test limit; zero uses all states.")
    parser.add_argument("--plastic-strain-threshold", type=float, default=1e-8)
    parser.add_argument("--regime-mode", choices=("relative", "absolute"), default="relative")
    parser.add_argument("--onset-fraction", type=float, default=0.05)
    parser.add_argument("--developed-fraction", type=float, default=0.50)
    parser.add_argument("--late-fraction", type=float, default=0.80)
    parser.add_argument("--minimum-regime-frames", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.noise_replicates < 0: parser.error("--noise-replicates must be non-negative")
    if len(args.window_lengths_mm) != 4: parser.error("Exactly four candidate physical windows are required.")
    return args


if __name__ == "__main__":
    main()
