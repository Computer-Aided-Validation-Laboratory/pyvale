"""Audit convergence and EGI window scales without running identification.

This diagnostic is deliberately truth-free for EGI selection: known maps are
used only to evaluate existing solve snapshots. Window diagnostics use the
phase-0 reference stress, stress sensitivities, fill fractions, and temporal
bootstrap stability.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter

from pyvale.vfm import EquilibriumGapMetric, ExperimentData, load_identification_result
from pyvale.vfm.equilibriumgapaggregation import calculate_nan_rms
from pyvale.vfm.metricsbvf import calculate_parameter_stress_sensitivities
from pyvale.vfm.postprocessing import (
    compute_plasticity_diagnostics,
    evaluate_snapshot_parameter_maps,
    load_constitutive_law_from_result,
    load_known_parameter_maps,
)


DATASET = Path(
    "/home/robh/1_Projects/pyvale-vfm-test-data/notched-ebw/"
    "synthetic-fe/wdbn1-idealised-yield/pyvale-vfm"
)
INPUT = DATASET / "prepared"
BASELINE_RESULT = (
    DATASET
    / "identification/prepared/spatial_weighting_baseline_20260827/"
    "identification_result.yaml"
)
OUTPUT = Path("dev/vfm/output/egi_window_convergence_checkpoint_20260827")
WINDOWS = (15, 17, 21, 29, 33, 41, 57)
BOOTSTRAP_SAMPLES = 48
RANDOM_SEED = 20260827


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)

    experiment_data = ExperimentData.load_from_file(_experiment_file(args.input))
    result = load_identification_result(args.result)
    law = load_constitutive_law_from_result(result)
    known_maps = load_known_parameter_maps(args.known_parameters, args.input)

    convergence = _audit_phase_one(
        experiment_data,
        result,
        law,
        known_maps,
    )
    _plot_convergence(convergence, figures / "phase_1_convergence.png")
    _plot_snapshot_validation(convergence, figures / "phase_1_snapshot_validation.png")
    _plot_basis_widths(convergence, figures / "phase_1_basis_widths.png")

    phase_zero_maps = _maps_from_snapshot(
        result.history.phases[0].final_snapshot,
        result.parameter_maps,
        experiment_data,
    )
    phase_zero_stress = law.calculate_stress(experiment_data.strain, phase_zero_maps)
    sensitivities = calculate_parameter_stress_sensitivities(
        experiment_data.strain,
        phase_zero_stress,
        law,
        phase_zero_maps,
        ("yield_strength", "hardening_modulus"),
    )
    window_diagnostics = _audit_windows(
        experiment_data,
        phase_zero_stress,
        sensitivities,
        args.windows,
        args.bootstrap_samples,
        args.random_seed,
    )
    _plot_window_maps(window_diagnostics, figures / "egi_window_maps_nominal.png")
    _plot_window_fill(window_diagnostics, figures / "egi_window_fill_fraction.png")
    _plot_window_summary(window_diagnostics, figures / "egi_window_summary.png")
    _plot_window_peaks(window_diagnostics, figures / "egi_window_peak_stability.png")
    _plot_normalisation_effect(
        window_diagnostics,
        figures / "egi_actual_vs_nominal_normalisation.png",
    )
    final_stress = law.calculate_stress(experiment_data.strain, result.parameter_maps)
    final_window_diagnostics = _audit_windows(
        experiment_data,
        final_stress,
        {},
        args.windows,
        args.bootstrap_samples,
        args.random_seed + 1,
    )
    aggregation = _audit_aggregation(
        experiment_data,
        window_diagnostics,
        final_window_diagnostics,
    )
    _plot_aggregation_comparison(
        aggregation,
        figures / "egi_aggregation_comparison.png",
    )

    dx, dy = _grid_spacing(experiment_data)
    summary = {
        "purpose": (
            "Bucket 0 and 1A diagnostic only; no identification optimisation "
            "was run. Known maps are used only for existing-snapshot validation."
        ),
        "input": str(_experiment_file(args.input)),
        "result": str(args.result),
        "grid": {
            "shape": list(experiment_data.specimen_geometry.x.shape),
            "dx_mm": dx,
            "dy_mm": dy,
            "x_span_mm": float(np.nanmax(experiment_data.specimen_geometry.x) - np.nanmin(experiment_data.specimen_geometry.x)),
            "y_span_mm": float(np.nanmax(experiment_data.specimen_geometry.y) - np.nanmin(experiment_data.specimen_geometry.y)),
        },
        "phase_1_convergence": convergence,
        "egi_window_diagnostics": _serialisable_window_diagnostics(window_diagnostics),
        "egi_aggregation_diagnostics": _serialisable_aggregation(aggregation),
        "interpretation_limits": [
            "Temporal bootstrap measures load-step stability, not an experimental DIC noise distribution.",
            "The nominal versus actual point-count comparison is diagnostic; it does not prescribe a normalization change.",
            "Window selection must not use the known parameter maps in this file.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "output": str(args.output),
        "phase_1_solves": len(convergence["solves"]),
        "windows": list(args.windows),
    }, indent=2))


def _audit_phase_one(
    experiment_data: ExperimentData,
    result: Any,
    law: Any,
    known_maps: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    known_yielded = None
    if known_maps is not None:
        plasticity = compute_plasticity_diagnostics(experiment_data, law, known_maps)
        known_yielded = None if plasticity is None else plasticity.yielded_datapoints

    solves = []
    for solve in result.history.phases[1].solve_results:
        snapshot = solve.final_snapshot
        maps = (
            None
            if snapshot is None
            else _maps_from_snapshot(snapshot, result.parameter_maps, experiment_data)
        )
        validation = _snapshot_validation(maps, known_maps, known_yielded)
        widths = [] if snapshot is None else _basis_widths(snapshot)
        history = solve.details.get("history", []) if solve.details else []
        costs = [float(item["cost"]) for item in history if "cost" in item]
        tail_reduction = None
        if len(costs) >= 6:
            tail_reduction = (costs[-6] - costs[-1]) / max(abs(costs[-6]), 1.0e-12)
        solves.append({
            "solve_iteration": int(solve.solve_iteration),
            "accepted": bool(solve.accepted),
            "status": str(solve.status),
            "message": solve.message,
            "num_evaluations": solve.num_evaluations,
            "runtime_seconds": solve.runtime_seconds,
            "final_cost": _finite_or_none(solve.final_objective.get("cost")),
            "history_iterations": len(history),
            "tail_relative_reduction_last_5": tail_reduction,
            "basis_count": len(widths),
            "basis_widths_mm": widths,
            **validation,
        })
    return {"solves": solves}


def _audit_windows(
    experiment_data: ExperimentData,
    reference_stress: np.ndarray,
    sensitivities: dict[str, Any],
    windows: tuple[int, ...],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    dx, dy = _grid_spacing(experiment_data)
    diagnostics: dict[str, Any] = {}
    nominal_maps: dict[int, np.ndarray] = {}

    for side in windows:
        metric = EquilibriumGapMetric(window_size=(side, side))
        metric.initialise(experiment_data)
        evaluation = metric.evaluate_equilibrium_gap(reference_stress)
        fields = evaluation.metric_result.additional_fields
        assert fields is not None
        nominal_map = np.asarray(fields["weighted_temporal_rms"], dtype=float)
        counts = np.asarray(fields["window_point_counts"], dtype=float)
        nominal_count = float(fields["nominal_window_point_count"])
        valid = np.asarray(fields["valid_centre_mask"], dtype=bool)
        fill = np.where(valid, counts / nominal_count, np.nan)
        actual_map = nominal_map * nominal_count / counts
        actual_map[~valid] = np.nan
        raw_gap = np.asarray(fields["normalised_gap"], dtype=float)
        temporal_weights = np.asarray(fields["temporal_weights"], dtype=float)
        peak_samples = _bootstrap_peaks(
            raw_gap,
            temporal_weights,
            valid,
            experiment_data,
            rng,
            bootstrap_samples,
        )
        sensitivity_p95 = {}
        for name, sensitivity in sensitivities.items():
            sensitivity_result = metric.evaluate_equilibrium_gap(sensitivity.total)
            sensitivity_map = np.asarray(
                sensitivity_result.weighted_temporal_rms,
                dtype=float,
            )
            sensitivity_p95[name] = _nanpercentile(sensitivity_map, 95.0)
        roughness = _roughness_proxy(nominal_map)
        nominal_peak = _map_peak(nominal_map, experiment_data)
        actual_peak = _map_peak(actual_map, experiment_data)
        diagnostics[str(side)] = {
            "side_points": side,
            "physical_span_x_mm": (side - 1) * dx,
            "physical_span_y_mm": (side - 1) * dy,
            "valid_centre_count": int(np.count_nonzero(valid)),
            "valid_centre_fraction_grid": float(np.mean(valid)),
            "fill_fraction_p05": _nanpercentile(fill, 5.0),
            "fill_fraction_median": _nanpercentile(fill, 50.0),
            "fill_fraction_p95": _nanpercentile(fill, 95.0),
            "nominal_peak": nominal_peak,
            "actual_count_peak": actual_peak,
            "normalisation_peak_shift_mm": float(np.hypot(
                nominal_peak["x_mm"] - actual_peak["x_mm"],
                nominal_peak["y_mm"] - actual_peak["y_mm"],
            )),
            "nominal_map_p95": _nanpercentile(nominal_map, 95.0),
            "actual_count_map_p95": _nanpercentile(actual_map, 95.0),
            "roughness_proxy": roughness,
            "sensitivity_activity_p95": sensitivity_p95,
            "temporal_bootstrap_peak_spread_mm": _peak_spread(peak_samples),
            "temporal_bootstrap_peak_samples": peak_samples.tolist(),
            "nominal_map": nominal_map,
            "actual_count_map": actual_map,
            "fill_map": fill,
        }
        nominal_maps[side] = nominal_map

    for side, map_value in nominal_maps.items():
        correlations = {
            str(other): _map_correlation(map_value, other_map)
            for other, other_map in nominal_maps.items()
            if other != side
        }
        diagnostics[str(side)]["nominal_map_correlations"] = correlations
    return diagnostics


def _audit_aggregation(
    experiment_data: ExperimentData,
    phase_zero: dict[str, Any],
    final: dict[str, Any],
) -> dict[str, Any]:
    """Compare local/global EGI map aggregation without changing optimisation.

    Baselines come from the phase-zero scalar RMS values, matching the phase-1
    objective's fixed-baseline convention.  The maximum aggregation is a
    placement diagnostic only; the running identification still uses the
    established length-weighted mean map.
    """

    required = ("17", "29", "57")
    missing = [key for key in required if key not in phase_zero or key not in final]
    if missing:
        return {"available": False, "missing_windows": missing}
    baselines = {
        key: calculate_nan_rms(np.asarray(phase_zero[key]["nominal_map"], dtype=float))
        for key in required
    }
    stages = {}
    for name, diagnostics in (("phase_zero", phase_zero), ("final", final)):
        scaled = {
            key: np.asarray(diagnostics[key]["nominal_map"], dtype=float) / baselines[key]
            for key in required
        }
        current = _weighted_map({key: scaled[key] for key in ("29", "57")})
        additive = _weighted_map(scaled)
        scale_maximum = _scale_balanced_maximum(scaled)
        stages[name] = {
            "current_29_57_map": current,
            "additive_17_29_57_map": additive,
            "scale_balanced_maximum_map": scale_maximum,
            "current_29_57_peak": _map_peak(current, experiment_data),
            "additive_17_29_57_peak": _map_peak(additive, experiment_data),
            "scale_balanced_maximum_peak": _map_peak(scale_maximum, experiment_data),
            "additive_vs_current_correlation": _map_correlation(additive, current),
            "maximum_vs_current_correlation": _map_correlation(scale_maximum, current),
            "additive_vs_current_top5_overlap": _top_region_overlap(additive, current),
            "maximum_vs_current_top5_overlap": _top_region_overlap(scale_maximum, current),
        }
    return {"available": True, "baselines": baselines, "stages": stages}


def _weighted_map(maps: dict[str, np.ndarray]) -> np.ndarray:
    weights = np.asarray([float(key) for key in maps], dtype=float)
    values = tuple(maps.values())
    numerator = np.nansum(
        [weight * value for weight, value in zip(weights, values, strict=True)],
        axis=0,
    )
    denominator = np.nansum(
        [weight * np.isfinite(value) for weight, value in zip(weights, values, strict=True)],
        axis=0,
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0.0,
    )


def _scale_balanced_maximum(maps: dict[str, np.ndarray]) -> np.ndarray:
    stacked = np.stack(tuple(maps.values()))
    valid = np.any(np.isfinite(stacked), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        maximum = np.nanmax(stacked, axis=0)
    maximum[~valid] = np.nan
    return maximum


def _top_region_overlap(first: np.ndarray, second: np.ndarray) -> float:
    first_valid = first[np.isfinite(first)]
    second_valid = second[np.isfinite(second)]
    if first_valid.size == 0 or second_valid.size == 0:
        return float("nan")
    first_top = first >= np.percentile(first_valid, 95.0)
    second_top = second >= np.percentile(second_valid, 95.0)
    union = first_top | second_top
    return float(np.count_nonzero(first_top & second_top) / np.count_nonzero(union))


def _maps_from_snapshot(snapshot: Any, fallback: dict[str, np.ndarray], experiment_data: ExperimentData) -> dict[str, np.ndarray]:
    if snapshot is None:
        raise ValueError("Expected a saved phase snapshot.")
    maps = {name: np.asarray(value, dtype=float).copy() for name, value in fallback.items()}
    maps.update(evaluate_snapshot_parameter_maps(snapshot, experiment_data))
    return maps


def _snapshot_validation(
    maps: dict[str, np.ndarray] | None,
    known_maps: dict[str, np.ndarray] | None,
    yielded: np.ndarray | None,
) -> dict[str, float | None]:
    if maps is None or known_maps is None:
        return {}
    known_yield = np.asarray(known_maps["yield_strength"], dtype=float)
    identified_yield = np.asarray(maps["yield_strength"], dtype=float)
    valid = np.isfinite(known_yield) & np.isfinite(identified_yield)
    if yielded is not None:
        valid &= yielded
    difference = identified_yield[valid] - known_yield[valid]
    percent = np.abs(difference / known_yield[valid]) * 100.0
    identified_h = float(np.nanmean(maps["hardening_modulus"]))
    known_h = float(np.nanmean(known_maps["hardening_modulus"]))
    return {
        "yielded_yield_rmse_mpa": float(np.sqrt(np.mean(difference**2))),
        "yielded_yield_mape_percent": float(np.mean(percent)),
        "hardening_mpa": identified_h,
        "hardening_absolute_error_mpa": abs(identified_h - known_h),
    }


def _basis_widths(snapshot: Any) -> list[dict[str, float]]:
    widths = []
    for parameterisations in snapshot.spatial_parameterisations.values():
        for parameterisation in parameterisations:
            summary = parameterisation.summary
            for kernel in summary.get("kernels", []):
                width = sorted(float(value) for value in kernel["width"])
                widths.append({"minor_mm": width[0], "major_mm": width[1]})
    return widths


def _bootstrap_peaks(
    normalised_gap: np.ndarray,
    temporal_weights: np.ndarray,
    valid: np.ndarray,
    experiment_data: ExperimentData,
    rng: np.random.Generator,
    samples: int,
) -> np.ndarray:
    peaks = np.empty((samples, 2), dtype=float)
    for sample_index in range(samples):
        indices = rng.integers(0, normalised_gap.shape[0], size=normalised_gap.shape[0])
        values = normalised_gap[indices]
        weights = temporal_weights[indices, None, None]
        weighted_squared = values**2 * weights
        valid_counts = np.sum(np.isfinite(weighted_squared), axis=0)
        weighted_sum = np.nansum(weighted_squared, axis=0)
        rms = np.full(valid.shape, np.nan, dtype=float)
        available = valid_counts > 0
        rms[available] = np.sqrt(weighted_sum[available] / valid_counts[available])
        rms[~valid] = np.nan
        peak = _map_peak(rms, experiment_data)
        peaks[sample_index] = (peak["x_mm"], peak["y_mm"])
    return peaks


def _map_peak(values: np.ndarray, experiment_data: ExperimentData) -> dict[str, float]:
    row, column = np.unravel_index(np.nanargmax(values), values.shape)
    return {
        "x_mm": float(experiment_data.specimen_geometry.x[row, column]),
        "y_mm": float(experiment_data.specimen_geometry.y[row, column]),
        "value": float(values[row, column]),
    }


def _roughness_proxy(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    smooth = uniform_filter(np.where(finite, values, 0.0), size=5)
    support = uniform_filter(finite.astype(float), size=5)
    smooth = np.divide(smooth, support, out=np.full_like(smooth, np.nan), where=support > 0.0)
    residual = values - smooth
    return float(1.4826 * np.nanmedian(np.abs(residual - np.nanmedian(residual))))


def _peak_spread(peaks: np.ndarray) -> float:
    centre = np.nanmedian(peaks, axis=0)
    return float(np.nanpercentile(np.hypot(peaks[:, 0] - centre[0], peaks[:, 1] - centre[1]), 95.0))


def _map_correlation(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    return float(np.corrcoef(first[valid], second[valid])[0, 1])


def _plot_convergence(audit: dict[str, Any], path: Path) -> None:
    solves = audit["solves"]
    x = [entry["solve_iteration"] + 1 for entry in solves]
    accepted = np.asarray([entry["accepted"] for entry in solves])
    costs = [entry["final_cost"] for entry in solves]
    evaluations = [entry["num_evaluations"] for entry in solves]
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.4), layout="constrained")
    axes[0].plot(x, costs, color="0.7", zorder=1)
    axes[0].scatter(np.asarray(x)[accepted], np.asarray(costs)[accepted], color="tab:green", label="accepted", zorder=2)
    axes[0].scatter(np.asarray(x)[~accepted], np.asarray(costs)[~accepted], color="tab:red", marker="x", label="rejected", zorder=2)
    axes[0].set(ylabel="Final objective", title="Phase-1 solve convergence", xticks=x)
    axes[0].legend()
    colours = ["tab:green" if value else "tab:red" for value in accepted]
    axes[1].bar(x, evaluations, color=colours)
    axes[1].axhline(5000, color="0.35", linestyle="--", label="configured max evaluations")
    axes[1].set(xlabel="Solve", ylabel="Objective evaluations", xticks=x)
    axes[1].legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_snapshot_validation(audit: dict[str, Any], path: Path) -> None:
    solves = audit["solves"]
    x = np.asarray([entry["solve_iteration"] + 1 for entry in solves])
    accepted = np.asarray([entry["accepted"] for entry in solves])
    metrics = (
        ("yielded_yield_rmse_mpa", "Yielded yield RMSE [MPa]"),
        ("yielded_yield_mape_percent", "Yielded yield MAPE [%]"),
        ("hardening_absolute_error_mpa", "Hardening absolute error [MPa]"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), layout="constrained")
    for axis, (key, label) in zip(axes, metrics, strict=True):
        values = np.asarray([entry.get(key, np.nan) for entry in solves])
        axis.plot(x, values, color="0.7")
        axis.scatter(x[accepted], values[accepted], color="tab:green")
        axis.scatter(x[~accepted], values[~accepted], color="tab:red", marker="x")
        axis.set(xlabel="Solve", ylabel=label, xticks=x)
    fig.suptitle("Known-map validation of saved phase-1 solve snapshots")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_basis_widths(audit: dict[str, Any], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.0, 4.2), layout="constrained")
    for entry in audit["solves"]:
        widths = entry["basis_widths_mm"]
        if not widths:
            continue
        solve = entry["solve_iteration"] + 1
        axis.scatter([solve] * len(widths), [item["major_mm"] for item in widths], color="tab:blue", label="major width" if solve == 1 else None)
        axis.scatter([solve] * len(widths), [item["minor_mm"] for item in widths], color="tab:orange", marker="x", label="minor width" if solve == 1 else None)
    axis.axhline(40.0, color="0.4", linestyle="--", label="ROI x-span")
    axis.set(yscale="log", xlabel="Solve", ylabel="Gaussian width [mm]", title="Basis widths in saved solve snapshots")
    axis.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_window_maps(diagnostics: dict[str, Any], path: Path) -> None:
    entries = list(diagnostics.values())
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.0), layout="constrained")
    for axis, entry in zip(axes.flat, entries, strict=False):
        image = axis.imshow(entry["nominal_map"], origin="lower", cmap="viridis")
        axis.set_title(f"{entry['side_points']} pt ({entry['physical_span_x_mm']:.1f} mm)")
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, shrink=0.72)
    for axis in axes.flat[len(entries):]:
        axis.set_visible(False)
    fig.suptitle("Phase-0 EGI temporal RMS, nominal-count normalization")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_window_fill(diagnostics: dict[str, Any], path: Path) -> None:
    entries = list(diagnostics.values())
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.0), layout="constrained")
    for axis, entry in zip(axes.flat, entries, strict=False):
        image = axis.imshow(entry["fill_map"], origin="lower", cmap="viridis", vmin=0.5, vmax=1.0)
        axis.set_title(f"{entry['side_points']} pt fill")
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, shrink=0.72)
    for axis in axes.flat[len(entries):]:
        axis.set_visible(False)
    fig.suptitle("Valid EGI-window fill fraction")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_window_summary(diagnostics: dict[str, Any], path: Path) -> None:
    entries = list(diagnostics.values())
    span = np.asarray([entry["physical_span_x_mm"] for entry in entries])
    valid = np.asarray([entry["valid_centre_count"] for entry in entries])
    roughness = np.asarray([entry["roughness_proxy"] for entry in entries])
    signal_yield = np.asarray([entry["sensitivity_activity_p95"]["yield_strength"] for entry in entries])
    signal_hardening = np.asarray([entry["sensitivity_activity_p95"]["hardening_modulus"] for entry in entries])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7), layout="constrained")
    axes[0].plot(span, valid, marker="o")
    axes[0].set(xlabel="Window span [mm]", ylabel="Valid centre count", title="Available EGI coverage")
    axes[1].plot(span, signal_yield / roughness, marker="o", label="yield-strength activity / roughness")
    axes[1].plot(span, signal_hardening / roughness, marker="s", label="hardening activity / roughness")
    axes[1].set(xlabel="Window span [mm]", ylabel="Diagnostic ratio", title="Sensitivity contrast proxy", yscale="log")
    axes[1].legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_window_peaks(diagnostics: dict[str, Any], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.0, 4.8), layout="constrained")
    for entry in diagnostics.values():
        samples = np.asarray(entry["temporal_bootstrap_peak_samples"])
        axis.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.35, label=f"{entry['side_points']} pt")
    axis.set(xlabel="x [mm]", ylabel="y [mm]", title="Temporal-bootstrap EGI peak locations")
    axis.legend(ncol=2, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_normalisation_effect(diagnostics: dict[str, Any], path: Path) -> None:
    selected = [key for key in ("17", "33", "29", "57") if key in diagnostics]
    fig, axes = plt.subplots(len(selected), 3, figsize=(10.0, 2.9 * len(selected)), layout="constrained")
    axes = np.atleast_2d(axes)
    for row, key in enumerate(selected):
        entry = diagnostics[key]
        nominal = entry["nominal_map"]
        actual = entry["actual_count_map"]
        ratio = actual / nominal
        for axis, values, title in zip(axes[row], (nominal, actual, ratio), ("Nominal count", "Actual count", "Actual / nominal"), strict=True):
            image = axis.imshow(values, origin="lower", cmap="viridis")
            axis.set(title=f"{entry['side_points']} pt: {title}", xticks=[], yticks=[])
            fig.colorbar(image, ax=axis, shrink=0.7)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_aggregation_comparison(aggregation: dict[str, Any], path: Path) -> None:
    if not aggregation["available"]:
        return
    figure, axes = plt.subplots(2, 3, figsize=(11.5, 6.6), layout="constrained")
    labels = (
        ("current_29_57_map", "29/57 weighted mean"),
        ("additive_17_29_57_map", "17/29/57 weighted mean"),
        ("scale_balanced_maximum_map", "17/29/57 scale-balanced maximum"),
    )
    for row, (stage, diagnostics) in enumerate(aggregation["stages"].items()):
        for axis, (key, title) in zip(axes[row], labels, strict=True):
            image = axis.imshow(diagnostics[key], origin="lower", cmap="viridis")
            axis.set(title=f"{stage.replace('_', ' ')}: {title}", xticks=[], yticks=[])
            figure.colorbar(image, ax=axis, shrink=0.75)
    figure.suptitle("EGI aggregation diagnostic; maximum map is not used by optimisation")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _grid_spacing(experiment_data: ExperimentData) -> tuple[float, float]:
    x = experiment_data.specimen_geometry.x
    y = experiment_data.specimen_geometry.y
    return (
        float(np.nanmedian(np.diff(x, axis=1))),
        float(np.nanmedian(np.diff(y, axis=0))),
    )


def _nanpercentile(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    return float("nan") if finite.size == 0 else float(np.percentile(finite, percentile))


def _finite_or_none(value: Any) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def _serialisable_window_diagnostics(
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Exclude rendered maps from the JSON summary; they are retained in plots."""

    return {
        key: {
            field: value
            for field, value in entry.items()
            if not field.endswith("_map") and not field.endswith("_samples")
        }
        for key, entry in diagnostics.items()
    }


def _serialisable_aggregation(aggregation: dict[str, Any]) -> dict[str, Any]:
    if not aggregation["available"]:
        return aggregation
    return {
        "available": True,
        "baselines": aggregation["baselines"],
        "stages": {
            name: {
                key: value
                for key, value in diagnostics.items()
                if not key.endswith("_map")
            }
            for name, diagnostics in aggregation["stages"].items()
        },
    }


def _experiment_file(path: Path) -> Path:
    return path / "experiment_data.yaml" if path.is_dir() else path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--result", type=Path, default=BASELINE_RESULT)
    parser.add_argument("--known-parameters", type=Path, default=INPUT / "known_parameter_maps.npz")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--windows", type=int, nargs="+", default=WINDOWS)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    if any(value < 3 or value % 2 == 0 for value in args.windows):
        parser.error("--windows must contain odd integers of at least 3.")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive.")
    args.windows = tuple(args.windows)
    return args


if __name__ == "__main__":
    main()
