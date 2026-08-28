"""Build an independent notched-EBW state library and score residual components.

This is an offline synthetic study. Production identification metrics and
objectives are not modified. Known material maps are used only to construct
controlled test states and evaluate discrimination.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
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
    HARDENING_BOUNDS,
    YIELD_BOUNDS,
    _active_masks,
    _basis_count,
    _complete_maps,
    _create_metrics,
)


REGIMES = {
    "pre_yield": np.arange(0, 3),
    "yield_onset": np.arange(3, 6),
    "developed_plasticity": np.arange(6, 10),
    "late_plasticity": np.arange(10, 14),
}
SUMMARIES = ("rms", "p90", "p95", "coherent_rms", "coherence_fraction")
METRIC_NAMES = ("egi29", "egi57", "fre")


@dataclass(slots=True)
class State:
    name: str
    source: str
    split: str
    family: str
    parameters: dict[str, object]
    maps: dict[str, np.ndarray]
    basis_count: int | None = None
    seed: int | None = None


def main() -> None:
    args = _parse_args()
    dataset = args.dataset.expanduser().resolve()
    campaign_root = args.campaign_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    prepared = dataset / "prepared"
    experiment = ExperimentData.load_from_file(prepared / "experiment_data.yaml")
    known_raw = load_known_parameter_maps(prepared / "known_parameter_maps.npz")
    if known_raw is None:
        raise RuntimeError("Known parameter maps are required.")
    known = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in known_raw.items()
    }
    reference_path = next(iter(sorted(
        campaign_root.glob("*/identification_result.yaml")
    )))
    reference = load_identification_result(reference_path)
    law = load_constitutive_law_from_result(reference)
    metrics = _create_metrics(experiment)
    mask = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(
        experiment.specimen_geometry.x, experiment.specimen_geometry.y
    )
    yielded, high_plastic = _active_masks(experiment, law, known, mask)

    independent = _independent_states(experiment, law, known, mask)
    optimiser = _optimiser_late_states(campaign_root, experiment, known)
    states = [*independent, *optimiser]
    _write_map_archive(output / "independent_state_maps.npz", independent)

    manifest_rows = []
    component_rows = []
    for index, state in enumerate(states, start=1):
        errors = _property_errors(state.maps, known, mask, yielded, high_plastic)
        manifest_rows.append({
            "name": state.name,
            "source": state.source,
            "split": state.split,
            "family": state.family,
            "parameters": json.dumps(state.parameters, sort_keys=True),
            "basis_count": state.basis_count,
            "seed": state.seed,
            **errors,
        })
        components = _component_scores(law, state.maps, experiment, metrics)
        component_rows.append({
            "name": state.name,
            "source": state.source,
            "split": state.split,
            "family": state.family,
            "basis_count": state.basis_count,
            "seed": state.seed,
            **errors,
            **components,
        })
        if index == 1 or index % 10 == 0 or index == len(states):
            print(f"[{index:3d}/{len(states)}] {state.name}", flush=True)

    _write_csv(output / "state_manifest.csv", manifest_rows)
    _write_csv(output / "component_scores.csv", component_rows)
    discrimination = _discrimination(component_rows)
    _write_csv(output / "component_discrimination.csv", discrimination)
    shortlist = _development_shortlist(component_rows, discrimination)
    (output / "component_shortlist.json").write_text(
        json.dumps(shortlist, indent=2), encoding="utf-8"
    )
    _plots(output, discrimination, shortlist)
    print(json.dumps({
        "output": str(output),
        "independent_states": len(independent),
        "development_states": sum(s.split == "development" for s in independent),
        "validation_states": sum(s.split == "validation" for s in independent),
        "optimiser_late_states": len(optimiser),
        "components": len(_component_names()),
        "shortlist": shortlist["selected_components"],
    }, indent=2))


def _independent_states(experiment, law, known, mask):
    x = np.asarray(experiment.specimen_geometry.x, dtype=np.float64)
    y = np.asarray(experiment.specimen_geometry.y, dtype=np.float64)
    truth = np.asarray(known["yield_strength"], dtype=np.float64)
    hardening = np.asarray(known["hardening_modulus"], dtype=np.float64)
    base = float(np.median(truth[mask]))
    contrast = truth - base
    positive = np.maximum(contrast, 0.0)
    centre = float(np.sum(x[mask] * positive[mask]) / np.sum(positive[mask]))
    maximum_contrast = float(np.max(positive[mask]))
    weld_feature = np.clip(positive / maximum_contrast, 0.0, 1.0)

    plasticity = compute_plasticity_diagnostics(experiment, law, known)
    if plasticity is None:
        raise RuntimeError("Plasticity diagnostics are unavailable.")
    peak = np.nanmax(
        np.asarray(plasticity.equivalent_plastic_strain, dtype=np.float64), axis=0
    )
    hot_spots = []
    for side in (y > 0.0, y < 0.0):
        selection = mask & side
        location = np.unravel_index(
            np.nanargmax(np.where(selection, peak, np.nan)), peak.shape
        )
        hot_spots.append((float(x[location]), float(y[location])))

    states = [State(
        name="reference_truth",
        source="independent",
        split="reference",
        family="truth",
        parameters={},
        maps={name: value.copy() for name, value in known.items()},
    )]

    def add(split, family, parameters, yield_map, hardening_map=None):
        count = sum(s.split == split and s.family == family for s in states)
        maps = {name: value.copy() for name, value in known.items()}
        maps["yield_strength"] = np.clip(yield_map, *YIELD_BOUNDS)
        if hardening_map is not None:
            maps["hardening_modulus"] = np.clip(hardening_map, *HARDENING_BOUNDS)
        states.append(State(
            name=f"{split}_{family}_{count:02d}",
            source="independent",
            split=split,
            family=family,
            parameters=parameters,
            maps=maps,
        ))

    designs = {
        "development": {
            "shift": (-1.2, -0.4, 0.4, 1.2),
            "width": (0.65, 0.85, 1.15, 1.35),
            "amplitude": (0.70, 0.90, 1.10, 1.30),
            "haz_amplitude": (-60.0, -25.0, 25.0, 60.0),
            "local_amplitude": (-60.0, -25.0, 25.0, 60.0),
            "compensation": ((-60.0, 1.20), (-25.0, 1.10), (25.0, 0.90), (60.0, 0.80)),
        },
        "validation": {
            "shift": (-2.0, -0.8, 0.8, 2.0),
            "width": (0.50, 0.75, 1.25, 1.50),
            "amplitude": (0.50, 0.80, 1.20, 1.50),
            "haz_amplitude": (-90.0, -40.0, 40.0, 90.0),
            "local_amplitude": (-90.0, -40.0, 40.0, 90.0),
            "compensation": ((-90.0, 1.30), (-40.0, 1.15), (40.0, 0.85), (90.0, 0.70)),
        },
    }
    for split, design in designs.items():
        for shift in design["shift"]:
            transformed = _transform_weld(
                truth, x, y, centre, base, shift=float(shift)
            )
            add(split, "location_shift", {"shift_mm": shift}, transformed)
        for width in design["width"]:
            transformed = _transform_weld(
                truth, x, y, centre, base, width=float(width)
            )
            add(split, "width_change", {"width_factor": width}, transformed)
        for amplitude in design["amplitude"]:
            transformed = base + float(amplitude) * contrast
            add(
                split,
                "amplitude_error",
                {"contrast_factor": amplitude},
                transformed,
            )
        for index, amplitude in enumerate(design["haz_amplitude"]):
            side = -1.0 if index % 2 == 0 else 1.0
            haz_centre = centre + side * 2.6
            feature = np.exp(-0.5 * ((x - haz_centre) / 0.65) ** 2)
            transformed = truth + float(amplitude) * feature
            add(
                split,
                "haz_band_error",
                {
                    "side": "left" if side < 0 else "right",
                    "amplitude_mpa": amplitude,
                    "sigma_x_mm": 0.65,
                },
                transformed,
            )
        for index, amplitude in enumerate(design["local_amplitude"]):
            hot_x, hot_y = hot_spots[index % 2]
            feature = np.exp(-0.5 * (
                ((x - hot_x) / 0.8) ** 2 + ((y - hot_y) / 0.6) ** 2
            ))
            transformed = truth + float(amplitude) * feature
            add(
                split,
                "high_plastic_local_error",
                {
                    "centre_x_mm": hot_x,
                    "centre_y_mm": hot_y,
                    "amplitude_mpa": amplitude,
                    "sigma_x_mm": 0.8,
                    "sigma_y_mm": 0.6,
                },
                transformed,
            )
        for yield_amplitude, hardening_factor in design["compensation"]:
            transformed = truth + float(yield_amplitude) * weld_feature
            hardening_map = hardening * float(hardening_factor)
            add(
                split,
                "hardening_compensation",
                {
                    "weld_yield_amplitude_mpa": yield_amplitude,
                    "hardening_factor": hardening_factor,
                },
                transformed,
                hardening_map,
            )

    combinations = {
        "development": (
            (-0.4, 0.85, 0.90, -25.0, 1.10),
            (0.4, 1.15, 1.10, 25.0, 0.90),
            (-1.2, 0.65, 1.30, -60.0, 1.20),
            (1.2, 1.35, 0.70, 60.0, 0.80),
            (-0.4, 1.35, 1.30, 25.0, 1.00),
            (0.4, 0.65, 0.70, -25.0, 1.00),
        ),
        "validation": (
            (-0.8, 0.75, 0.80, -40.0, 1.15),
            (0.8, 1.25, 1.20, 40.0, 0.85),
            (-2.0, 0.50, 1.50, -90.0, 1.30),
            (2.0, 1.50, 0.50, 90.0, 0.70),
            (-0.8, 1.50, 1.50, 40.0, 1.00),
            (0.8, 0.50, 0.50, -40.0, 1.00),
        ),
    }
    for split, rows in combinations.items():
        for shift, width, amplitude, local_amplitude, hardening_factor in rows:
            transformed = _transform_weld(
                truth,
                x,
                y,
                centre,
                base,
                shift=shift,
                width=width,
                amplitude=amplitude,
            )
            hot_features = [
                np.exp(-0.5 * (
                    ((x - hot_x) / 0.8) ** 2
                    + ((y - hot_y) / 0.6) ** 2
                ))
                for hot_x, hot_y in hot_spots
            ]
            transformed += local_amplitude * np.maximum.reduce(hot_features)
            add(
                split,
                "combined_error",
                {
                    "shift_mm": shift,
                    "width_factor": width,
                    "contrast_factor": amplitude,
                    "local_amplitude_mpa": local_amplitude,
                    "hardening_factor": hardening_factor,
                },
                transformed,
                hardening * hardening_factor,
            )
    return states


def _transform_weld(
    truth,
    x,
    y,
    centre,
    base,
    *,
    shift=0.0,
    width=1.0,
    amplitude=1.0,
):
    x_axis = np.asarray(x[0, :], dtype=np.float64)
    y_axis = np.asarray(y[:, 0], dtype=np.float64)
    interpolator = RegularGridInterpolator(
        (y_axis, x_axis), truth, bounds_error=False, fill_value=base
    )
    source_x = centre + (x - centre - shift) / width
    points = np.column_stack((y.ravel(), source_x.ravel()))
    shifted = interpolator(points).reshape(truth.shape)
    return base + amplitude * (shifted - base)


def _optimiser_late_states(campaign_root, experiment, known):
    states = []
    # Use zero-gate trajectories only; 0.5% duplicates most maps.
    pattern = "spd_sensitivity_gate0p0pct_seed*/identification_result.yaml"
    for result_path in sorted(campaign_root.glob(pattern)):
        result = load_identification_result(result_path)
        seed = int(result_path.parent.name.rsplit("seed", 1)[1])
        for solve in result.history.phases[-1].solve_results:
            if solve.final_snapshot is None:
                continue
            basis_count = _basis_count(solve.final_snapshot)
            if basis_count < 5 or not solve.accepted:
                continue
            maps = _complete_maps(
                evaluate_snapshot_parameter_maps(
                    solve.final_snapshot, experiment
                ),
                known,
            )
            states.append(State(
                name=f"optimiser_seed{seed:02d}_bf{basis_count}",
                source="optimiser_late",
                split="diagnostic",
                family="optimiser_bf5_8",
                parameters={"seed": seed, "basis_count": basis_count},
                maps=maps,
                basis_count=basis_count,
                seed=seed,
            ))
    return states


def _property_errors(maps, known, mask, yielded, high_plastic):
    yield_error = maps["yield_strength"] - known["yield_strength"]
    hardening_error = maps["hardening_modulus"] - known["hardening_modulus"]
    return {
        "roi_rmse_mpa": _rmse(yield_error, mask),
        "yielded_rmse_mpa": _rmse(yield_error, yielded),
        "high_plastic_rmse_mpa": _rmse(yield_error, high_plastic),
        "hardening_rmse_mpa": _rmse(hardening_error, mask),
    }


def _rmse(error, selection):
    return float(np.sqrt(np.mean(np.asarray(error)[selection] ** 2)))


def _component_scores(law, maps, experiment, metrics):
    stress = law.calculate_stress(experiment.strain, maps)
    force_metric, egi_metrics = metrics
    fields = []
    temporal_weights = []
    spatial_weights = []
    for metric in egi_metrics:
        result = metric.evaluate_equilibrium_gap(stress)
        fields.append(np.asarray(result.normalised_gap, dtype=np.float64))
        temporal_weights.append(np.asarray(
            result.metric_result.additional_fields["temporal_weights"],
            dtype=np.float64,
        ))
        spatial_weights.append(None)
    force_result = force_metric.evaluate_force_recon_error(stress, experiment)
    metadata = force_result.metric_result.additional_fields
    fields.append(
        np.asarray(metadata["normalised_residual"], dtype=np.float64)
    )
    temporal_weights.append(
        np.asarray(metadata["temporal_weights"], dtype=np.float64)
    )
    spatial_weights.append(
        np.asarray(metadata["spatial_weights"], dtype=np.float64)
    )

    output = {}
    for metric_name, field, temporal, spatial in zip(
        METRIC_NAMES, fields, temporal_weights, spatial_weights, strict=True
    ):
        for regime_name, indices in REGIMES.items():
            values = field[indices]
            time_weights = temporal[indices]
            rms = _weighted_rms(values, time_weights, spatial)
            coherent = _coherent_rms(values, time_weights, spatial)
            prefix = f"{metric_name}__{regime_name}"
            output[f"{prefix}__rms"] = rms
            output[f"{prefix}__p90"] = _tail_percentile(
                values, time_weights, spatial, 0.90
            )
            output[f"{prefix}__p95"] = _tail_percentile(
                values, time_weights, spatial, 0.95
            )
            output[f"{prefix}__coherent_rms"] = coherent
            output[f"{prefix}__coherence_fraction"] = coherent / max(
                rms, np.finfo(float).eps
            )
    return output


def _normalise(weights):
    values = np.asarray(weights, dtype=np.float64)
    total = float(np.nansum(values))
    if total > 0.0:
        return values / total
    return np.full(values.shape, 1.0 / values.size)


def _observation_weights(field, temporal, spatial):
    values = np.asarray(field)
    time = _normalise(temporal)
    if values.ndim == 3:
        weights = np.broadcast_to(
            time[:, None, None], values.shape
        ).copy()
    else:
        space = _normalise(spatial)
        weights = time[:, None] * space[None, :]
    weights[~np.isfinite(values)] = 0.0
    return _normalise(weights)


def _weighted_rms(field, temporal, spatial):
    weights = _observation_weights(field, temporal, spatial)
    return float(np.sqrt(np.nansum(weights * np.asarray(field) ** 2)))


def _weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    target = quantile * cumulative[-1]
    index = min(np.searchsorted(cumulative, target), values.size - 1)
    return float(values[order[index]])


def _tail_percentile(field, temporal, spatial, quantile):
    weights = _observation_weights(field, temporal, spatial)
    return _weighted_quantile(np.abs(field), weights, quantile)


def _coherent_rms(field, temporal, spatial):
    values = np.asarray(field, dtype=np.float64)
    finite = np.isfinite(values)
    filled = np.nan_to_num(values, nan=0.0)
    if values.ndim == 3:
        sigma = (0.0, 2.0, 2.0)
        coherent = gaussian_filter(filled, sigma=sigma, mode="nearest")
        support = gaussian_filter(
            finite.astype(float), sigma=sigma, mode="nearest"
        )
    else:
        coherent = gaussian_filter1d(
            filled, sigma=2.0, axis=1, mode="nearest"
        )
        support = gaussian_filter1d(
            finite.astype(float), sigma=2.0, axis=1, mode="nearest"
        )
    coherent = np.divide(
        coherent,
        support,
        out=np.full_like(coherent, np.nan),
        where=support > 0.25,
    )
    return _weighted_rms(coherent, temporal, spatial)


def _component_names():
    return [
        f"{metric}__{regime}__{summary}"
        for metric in METRIC_NAMES
        for regime in REGIMES
        for summary in SUMMARIES
    ]


def _discrimination(rows):
    subsets = {
        "development": [
            row for row in rows if row["split"] == "development"
        ],
        "validation": [
            row for row in rows if row["split"] == "validation"
        ],
        "optimiser_bf5_8": [
            row for row in rows if row["source"] == "optimiser_late"
        ],
    }
    output = []
    for subset_name, selected in subsets.items():
        for target in ("yielded_rmse_mpa", "high_plastic_rmse_mpa"):
            errors = np.asarray(
                [float(row[target]) for row in selected], dtype=np.float64
            )
            for component in _component_names():
                scores = np.asarray(
                    [float(row[component]) for row in selected],
                    dtype=np.float64,
                )
                coefficient, p_value = spearmanr(scores, errors)
                pairwise = _pairwise_statistics(scores, errors)
                output.append({
                    "subset": subset_name,
                    "target": target,
                    "component": component,
                    "metric": component.split("__")[0],
                    "regime": component.split("__")[1],
                    "summary": component.split("__")[2],
                    "states": len(selected),
                    "spearman_r": float(coefficient),
                    "p_value": float(p_value),
                    "pairwise_accuracy": pairwise["accuracy"],
                    "pairwise_pairs": pairwise["pairs"],
                    "pairwise_coverage": pairwise["coverage"],
                })
    return output


def _pairwise_statistics(scores, errors):
    score_delta = scores[:, None] - scores[None, :]
    error_delta = errors[:, None] - errors[None, :]
    upper = np.triu(np.ones(score_delta.shape, dtype=bool), k=1)
    valid = upper & (score_delta != 0.0) & (error_delta != 0.0)
    total = int(np.count_nonzero(upper & (error_delta != 0.0)))
    pairs = int(np.count_nonzero(valid))
    accuracy = (
        float(np.mean(
            np.sign(score_delta[valid]) == np.sign(error_delta[valid])
        ))
        if pairs
        else float("nan")
    )
    return {
        "accuracy": accuracy,
        "pairs": pairs,
        "coverage": pairs / total if total else float("nan"),
    }


def _development_shortlist(rows, discrimination):
    lookup = {
        (row["subset"], row["target"], row["component"]): row
        for row in discrimination
    }
    components = _component_names()

    # Carry-forward eligibility is fixed before inspecting optimiser-late
    # rankings. Development and validation must both show broad, strong
    # discrimination for both error targets. Validation is pass/fail only;
    # no aggregation weight or threshold is tuned on it.
    thresholds = {
        "spearman_r": 0.75,
        "pairwise_accuracy": 0.75,
        "pairwise_coverage": 0.80,
    }
    eligible = []
    for component in components:
        passes = True
        for subset in ("development", "validation"):
            for target in ("yielded_rmse_mpa", "high_plastic_rmse_mpa"):
                result = lookup[(subset, target, component)]
                passes &= all(
                    np.isfinite(result[field])
                    and result[field] >= threshold
                    for field, threshold in thresholds.items()
                )
        if passes:
            eligible.append(component)

    # From independently confirmed components, retain the best BF5-8
    # yielded-error and high-plastic-error evidence separately. This is a
    # diagnostic shortlist, not a combined scalar.
    selected = []
    selection_records = []
    for target in ("yielded_rmse_mpa", "high_plastic_rmse_mpa"):
        ranked = []
        for component in eligible:
            result = lookup[("optimiser_bf5_8", target, component)]
            signed_pairwise = 2.0 * result["pairwise_accuracy"] - 1.0
            utility = min(result["spearman_r"], signed_pairwise)
            ranked.append((utility, result["spearman_r"], component))
        utility, coefficient, component = max(ranked)
        if component not in selected:
            selected.append(component)
        selection_records.append({
            "target": target,
            "component": component,
            "optimiser_late_utility": utility,
            "optimiser_late_spearman": coefficient,
        })

    # Preserve one local-error specialist. These small spatial perturbations
    # are deliberately rare in the broad library and are a named failure mode,
    # so choose on development local cases and only report validation later.
    local_development = [
        row for row in rows
        if row["split"] == "development"
        and row["family"] == "high_plastic_local_error"
    ]
    local_errors = np.asarray([
        float(row["high_plastic_rmse_mpa"]) for row in local_development
    ])
    local_ranked = []
    for component in eligible:
        scores = np.asarray([
            float(row[component]) for row in local_development
        ])
        coefficient = float(spearmanr(scores, local_errors).statistic)
        pairwise = _pairwise_statistics(scores, local_errors)
        signed_pairwise = 2.0 * pairwise["accuracy"] - 1.0
        utility = min(coefficient, signed_pairwise)
        local_ranked.append((utility, coefficient, component))
    utility, coefficient, component = max(local_ranked)
    if component not in selected:
        selected.append(component)
    selection_records.append({
        "target": "high_plastic_local_error",
        "component": component,
        "development_utility": utility,
        "development_spearman": coefficient,
    })

    evidence = {}
    for component in selected:
        evidence[component] = {}
        for subset in ("development", "validation", "optimiser_bf5_8"):
            evidence[component][subset] = {}
            for target in ("yielded_rmse_mpa", "high_plastic_rmse_mpa"):
                row = lookup[(subset, target, component)]
                evidence[component][subset][target] = {
                    "spearman_r": row["spearman_r"],
                    "pairwise_accuracy": row["pairwise_accuracy"],
                    "states": row["states"],
                }
    complementarity = _component_complementarity(rows, selected)
    return {
        "selection_protocol": (
            "Require development and validation Spearman rho, pairwise "
            "accuracy, and pairwise coverage all >= 0.75, 0.75, and 0.80 "
            "for both property-error targets. Among eligible components, "
            "retain the strongest BF5-8 yielded-error and high-plastic-error "
            "diagnostics separately. No weights are fitted."
        ),
        "eligibility_thresholds": thresholds,
        "eligible_components": eligible,
        "selection_records": selection_records,
        "selected_components": selected,
        "complementarity": complementarity,
        "held_out_evidence": evidence,
    }


def _component_complementarity(rows, selected):
    if len(selected) < 2:
        return {}
    first, second = selected[:2]
    output = {}
    for subset, chosen in {
        "development": [row for row in rows if row["split"] == "development"],
        "validation": [row for row in rows if row["split"] == "validation"],
        "optimiser_bf5_8": [row for row in rows if row["source"] == "optimiser_late"],
    }.items():
        first_scores = np.asarray([float(row[first]) for row in chosen])
        second_scores = np.asarray([float(row[second]) for row in chosen])
        output[subset] = {
            "score_spearman": float(
                spearmanr(first_scores, second_scores).statistic
            )
        }
        for target in ("yielded_rmse_mpa", "high_plastic_rmse_mpa"):
            errors = np.asarray([float(row[target]) for row in chosen])
            first_correct, first_valid = _pairwise_correct(first_scores, errors)
            second_correct, second_valid = _pairwise_correct(second_scores, errors)
            first_misses = first_valid & second_valid & ~first_correct
            second_misses = first_valid & second_valid & ~second_correct
            output[subset][target] = {
                "second_rescues_first_misses": (
                    float(np.mean(second_correct[first_misses]))
                    if np.any(first_misses) else 0.0
                ),
                "first_rescues_second_misses": (
                    float(np.mean(first_correct[second_misses]))
                    if np.any(second_misses) else 0.0
                ),
            }
    return output


def _pairwise_correct(scores, errors):
    score_delta = scores[:, None] - scores[None, :]
    error_delta = errors[:, None] - errors[None, :]
    upper = np.triu(np.ones(score_delta.shape, dtype=bool), k=1)
    valid = upper & (score_delta != 0.0) & (error_delta != 0.0)
    correct = np.zeros(score_delta.shape, dtype=bool)
    correct[valid] = np.sign(score_delta[valid]) == np.sign(error_delta[valid])
    return correct, valid


def _plots(output, discrimination, shortlist):
    for target, label in (
        ("yielded_rmse_mpa", "Yielded-region error"),
        ("high_plastic_rmse_mpa", "High-plastic-region error"),
    ):
        figure, axes = plt.subplots(
            1, 3, figsize=(14, 5), layout="constrained", sharey=True
        )
        for axis, subset in zip(
            axes,
            ("development", "validation", "optimiser_bf5_8"),
            strict=True,
        ):
            rows = [
                row for row in discrimination
                if row["subset"] == subset and row["target"] == target
            ]
            matrix = np.full(
                (len(METRIC_NAMES) * len(SUMMARIES), len(REGIMES)), np.nan
            )
            y_labels = []
            for metric_index, metric in enumerate(METRIC_NAMES):
                for summary_index, summary in enumerate(SUMMARIES):
                    row_index = metric_index * len(SUMMARIES) + summary_index
                    y_labels.append(f"{metric} / {summary}")
                    for regime_index, regime in enumerate(REGIMES):
                        match = next(
                            row for row in rows
                            if row["metric"] == metric
                            and row["summary"] == summary
                            and row["regime"] == regime
                        )
                        matrix[row_index, regime_index] = match["spearman_r"]
            image = axis.imshow(
                matrix,
                aspect="auto",
                cmap="RdYlBu",
                vmin=-1.0,
                vmax=1.0,
            )
            axis.set_title(subset.replace("_", " "))
            axis.set_xticks(
                np.arange(len(REGIMES)),
                [name.replace("_", "\n") for name in REGIMES],
                rotation=25,
                ha="right",
            )
            axis.set_yticks(np.arange(len(y_labels)), y_labels)
        figure.colorbar(image, ax=axes, label="Spearman correlation")
        figure.suptitle(f"Component discrimination: {label}")
        figure.savefig(
            output / f"component_heatmap_{target}.png", dpi=180
        )
        plt.close(figure)

    selected = shortlist["selected_components"]
    figure, axes = plt.subplots(
        1, 2, figsize=(11, 4.5), layout="constrained"
    )
    for axis, target in zip(
        axes,
        ("yielded_rmse_mpa", "high_plastic_rmse_mpa"),
        strict=True,
    ):
        x = np.arange(len(selected))
        width = 0.25
        for offset, subset in enumerate(
            ("development", "validation", "optimiser_bf5_8")
        ):
            values = [
                next(
                    row["spearman_r"] for row in discrimination
                    if row["subset"] == subset
                    and row["target"] == target
                    and row["component"] == component
                )
                for component in selected
            ]
            axis.bar(
                x + (offset - 1) * width,
                values,
                width,
                label=subset.replace("_", " "),
            )
        axis.set_xticks(x, selected, rotation=25, ha="right")
        axis.set_ylim(-1.0, 1.0)
        axis.axhline(0.0, color="black", lw=0.7)
        axis.set_title(target.replace("_mpa", "").replace("_", " "))
        axis.set_ylabel("Spearman correlation")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    figure.savefig(output / "shortlist_validation.png", dpi=180)
    plt.close(figure)


def _write_map_archive(path, states):
    independent = [state for state in states if state.split != "reference"]
    np.savez_compressed(
        path,
        names=np.asarray([state.name for state in independent]),
        splits=np.asarray([state.split for state in independent]),
        families=np.asarray([state.family for state in independent]),
        yield_strength=np.stack([
            state.maps["yield_strength"] for state in independent
        ]),
        hardening_modulus=np.stack([
            state.maps["hardening_modulus"] for state in independent
        ]),
    )


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
