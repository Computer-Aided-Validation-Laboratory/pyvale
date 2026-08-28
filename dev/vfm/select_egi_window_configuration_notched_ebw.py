"""Select EGI window and peak-smoothing settings without identification solves.

The selection metrics deliberately use only specimen geometry, EGI fields and
perturbation stability.  Known parameter maps are not loaded.  This makes the
procedure transferable: a later experimental run can replace the dimensionless
noise sweep with its measured DIC noise ratio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter

from pyvale.vfm import EquilibriumGapMetric, ExperimentData, load_identification_result
from pyvale.vfm.postprocessing import (
    evaluate_snapshot_parameter_maps,
    load_constitutive_law_from_result,
)


DATASET = Path(
    "/home/robh/1_Projects/pyvale-vfm-test-data/notched-ebw/"
    "synthetic-fe/wdbn1-idealised-yield/pyvale-vfm"
)
INPUT = DATASET / "prepared"
RESULT = DATASET / "identification/prepared/spatial_weighting_baseline_20260827/identification_result.yaml"
OUTPUT = Path("dev/vfm/output/egi_window_selection_20260827")
FRACTIONS = (0.125, 0.25, 0.5, 1.0)
NOISE_RATIOS = (0.0, 0.0025, 0.005)
BOOTSTRAP_SAMPLES = 48
NOISE_SAMPLES = 12
SEED = 20260827


@dataclass(frozen=True)
class Window:
    fraction: float
    shape: tuple[int, int]
    span_x_mm: float
    span_y_mm: float

    @property
    def label(self) -> str:
        return f"{self.shape[0]}x{self.shape[1]}"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figures = OUTPUT / "figures"
    figures.mkdir(exist_ok=True)

    experiment = ExperimentData.load_from_file(INPUT / "experiment_data.yaml")
    result = load_identification_result(RESULT)
    law = load_constitutive_law_from_result(result)
    maps = _phase_zero_maps(result, experiment)
    reference_stress = law.calculate_stress(experiment.strain, maps)
    dx, dy = _grid_spacing(experiment)
    lref = min(_span(experiment.specimen_geometry.x), _span(experiment.specimen_geometry.y))
    windows = tuple(_physical_window(fraction, lref, dx, dy) for fraction in FRACTIONS)
    metrics = {window.label: _initialised_metric(window, experiment) for window in windows}

    reference = {
        label: _evaluate(metric, reference_stress, experiment, BOOTSTRAP_SAMPLES, np.random.default_rng(SEED + index))
        for index, (label, metric) in enumerate(metrics.items())
    }
    noise = _noise_stability(experiment, law, maps, metrics, reference, lref)
    diagnostics = _merge_diagnostics(windows, reference, noise)
    selected = _select_windows(diagnostics, lref)
    smoothing = _select_smoothing(selected, reference, experiment, lref)

    _plot_scale_diagnostics(diagnostics, lref, figures / "scale_diagnostics.png")
    _plot_peak_locations(diagnostics, figures / "peak_locations.png")
    _plot_noise_stability(diagnostics, lref, figures / "noise_stability.png")
    _plot_smoothing(smoothing, lref, figures / "smoothing_selection.png")
    _plot_selected_maps(selected, reference, smoothing["selected_points"], figures / "selected_configuration_maps.png")

    summary = {
        "purpose": "Bucket 1C-1F metric-only EGI configuration selection; no identification optimisation was run.",
        "selection_principles": [
            "Nominal window-count normalization is retained.",
            "Window scale is expressed as a fraction of the smaller physical ROI dimension.",
            "Known parameter maps are not used for selecting windows or smoothing.",
            "The production configuration contains at most a local and a global window.",
        ],
        "grid": {"dx_mm": dx, "dy_mm": dy, "reference_length_mm": lref},
        "noise_ratios": list(NOISE_RATIOS),
        "window_diagnostics": diagnostics,
        "selection": selected,
        "smoothing": smoothing,
        "limitations": [
            "The noise sweep is independent Gaussian strain noise expressed relative to loaded strain RMS; it is not a measured DIC covariance model.",
            "Temporal bootstrap tests load-history sensitivity, not independent experimental repeats.",
        ],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_report(summary, OUTPUT / "REPORT.md")
    print(json.dumps({"output": str(OUTPUT), "selection": selected, "smoothing": smoothing}, indent=2))


def _phase_zero_maps(result: object, experiment: ExperimentData) -> dict[str, np.ndarray]:
    snapshot = result.history.phases[0].final_snapshot
    if snapshot is None:
        raise ValueError("The saved identification result has no phase-0 snapshot.")
    maps = {name: np.asarray(value, dtype=float).copy() for name, value in result.parameter_maps.items()}
    maps.update(evaluate_snapshot_parameter_maps(snapshot, experiment))
    return maps


def _physical_window(fraction: float, lref: float, dx: float, dy: float) -> Window:
    target = fraction * lref
    shape = (_nearest_odd_points(target, dy), _nearest_odd_points(target, dx))
    return Window(
        fraction=fraction,
        shape=shape,
        span_x_mm=(shape[1] - 1) * dx,
        span_y_mm=(shape[0] - 1) * dy,
    )


def _nearest_odd_points(length: float, spacing: float) -> int:
    raw = length / spacing + 1.0
    candidates = [value for value in range(3, max(5, int(np.ceil(raw)) + 3), 2)]
    return min(candidates, key=lambda value: abs(value - raw))


def _initialised_metric(window: Window, experiment: ExperimentData) -> EquilibriumGapMetric:
    metric = EquilibriumGapMetric(window_size=window.shape)
    metric.initialise(experiment)
    return metric


def _evaluate(
    metric: EquilibriumGapMetric,
    stress: np.ndarray,
    experiment: ExperimentData,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    evaluation = metric.evaluate_equilibrium_gap(stress)
    fields = evaluation.metric_result.additional_fields
    assert fields is not None
    values = np.asarray(fields["weighted_temporal_rms"], dtype=float)
    valid = np.asarray(fields["valid_centre_mask"], dtype=bool)
    counts = np.asarray(fields["window_point_counts"], dtype=float)
    nominal_count = float(fields["nominal_window_point_count"])
    raw_gap = np.asarray(fields["normalised_gap"], dtype=float)
    temporal_weights = np.asarray(fields["temporal_weights"], dtype=float)
    peaks = _bootstrap_peaks(raw_gap, temporal_weights, valid, experiment, rng, samples)
    return {
        "map": values,
        "valid": valid,
        "fill": np.where(valid, counts / nominal_count, np.nan),
        "temporal_peaks": peaks,
        "temporal_spread_mm": _peak_spread(peaks),
        "peak": _peak(values, experiment),
        "map_rms": _nan_rms(values),
        "raw_gap": raw_gap,
        "temporal_weights": temporal_weights,
    }


def _noise_stability(
    experiment: ExperimentData,
    law: object,
    maps: dict[str, np.ndarray],
    metrics: dict[str, EquilibriumGapMetric],
    reference: dict[str, dict[str, object]],
    lref: float,
) -> dict[str, dict[str, dict[str, float]]]:
    rng = np.random.default_rng(SEED + 100)
    signal_rms = _nan_rms(experiment.strain)
    peaks: dict[str, dict[str, list[list[float]]]] = {
        label: {str(ratio): [] for ratio in NOISE_RATIOS} for label in metrics
    }
    for label, entry in reference.items():
        peaks[label]["0.0"] = np.asarray(entry["temporal_peaks"], dtype=float).tolist()
    finite = np.isfinite(experiment.strain)
    for ratio in NOISE_RATIOS[1:]:
        sigma = ratio * signal_rms
        for _ in range(NOISE_SAMPLES):
            noise = rng.normal(0.0, sigma, size=experiment.strain.shape)
            noisy_strain = np.where(finite, experiment.strain + noise, np.nan)
            stress = law.calculate_stress(noisy_strain, maps)
            for label, metric in metrics.items():
                evaluation = metric.evaluate_equilibrium_gap(stress)
                values = np.asarray(evaluation.metric_result.additional_fields["weighted_temporal_rms"], dtype=float)
                values = _smooth_nan(values, 1)
                peak = _peak(values, experiment)
                peaks[label][str(ratio)].append([peak["x_mm"], peak["y_mm"]])
    return {
        label: {
            ratio: {
                "peak_spread_mm": _peak_spread(np.asarray(samples, dtype=float)),
                "peak_spread_fraction_lref": _peak_spread(np.asarray(samples, dtype=float)) / lref,
            }
            for ratio, samples in by_ratio.items()
        }
        for label, by_ratio in peaks.items()
    }


def _merge_diagnostics(
    windows: tuple[Window, ...],
    reference: dict[str, dict[str, object]],
    noise: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, object]]:
    diagnostics: dict[str, dict[str, object]] = {}
    for window in windows:
        entry = reference[window.label]
        correlations = {
            other: _correlation(np.asarray(entry["map"]), np.asarray(other_entry["map"]))
            for other, other_entry in reference.items() if other != window.label
        }
        diagnostics[window.label] = {
            "fraction_of_reference_length": window.fraction,
            "shape_points": list(window.shape),
            "span_x_mm": window.span_x_mm,
            "span_y_mm": window.span_y_mm,
            "fill_p05": _percentile(np.asarray(entry["fill"]), 5.0),
            "valid_centre_fraction": float(np.mean(np.asarray(entry["valid"]))),
            "temporal_peak_spread_mm": entry["temporal_spread_mm"],
            "peak": entry["peak"],
            "noise": noise[window.label],
            "map_correlations": correlations,
        }
    return diagnostics


def _select_windows(diagnostics: dict[str, dict[str, object]], lref: float) -> dict[str, object]:
    ordered = sorted(diagnostics, key=lambda key: float(diagnostics[key]["fraction_of_reference_length"]))
    # One-quarter and one-half of the smaller ROI dimension are the smallest
    # production scales.  The one-eighth scale remains a resolution diagnostic.
    production = [key for key in ordered if float(diagnostics[key]["fraction_of_reference_length"]) >= 0.25]
    stable = [
        key for key in production
        if float(diagnostics[key]["fill_p05"]) >= 0.5
        and max(float(item["peak_spread_fraction_lref"]) for item in diagnostics[key]["noise"].values()) <= 0.10
    ]
    local = stable[0] if stable else production[0]
    global_candidates = [key for key in stable if key != local]
    global_window = next(
        (key for key in global_candidates if float(diagnostics[local]["map_correlations"][key]) < 0.98),
        None,
    )
    return {
        "local_window": local,
        "global_window": global_window,
        "windows": [value for value in (local, global_window) if value is not None],
        "stable_candidates": stable,
        "criteria": {
            "minimum_production_fraction": 0.25,
            "minimum_fill_fraction_p05": 0.5,
            "maximum_peak_spread_fraction_lref": 0.10,
            "maximum_redundant_map_correlation": 0.98,
        },
    }


def _select_smoothing(
    selected: dict[str, object],
    reference: dict[str, dict[str, object]],
    experiment: ExperimentData,
    lref: float,
) -> dict[str, object]:
    labels = list(selected["windows"])
    smallest = min(int(label.split("x")[0]) for label in labels)
    candidates = sorted({1, _odd_round(0.10 * smallest), _odd_round(0.20 * smallest)})
    weighted = _combined_map(labels, reference)
    rng = np.random.default_rng(SEED + 200)
    bootstrap_maps = [_bootstrap_map_for_labels(labels, reference, experiment, rng) for _ in range(BOOTSTRAP_SAMPLES)]
    entries = []
    for points in candidates:
        peaks = np.asarray([list(_peak(_smooth_nan(values, points), experiment).values())[:2] for values in bootstrap_maps])
        entries.append({"points": points, "physical_width_mm": (points - 1) * min(_grid_spacing(experiment)), "peak_spread_mm": _peak_spread(peaks), "peak_spread_fraction_lref": _peak_spread(peaks) / lref})
    spreads = np.asarray([entry["peak_spread_mm"] for entry in entries])
    selected_index = 0
    for index in range(len(entries) - 1):
        improvement = (spreads[index] - spreads[index + 1]) / max(spreads[index], 1.0e-12)
        if improvement < 0.10:
            selected_index = index
            break
    else:
        selected_index = int(np.argmin(spreads))
    return {"entries": entries, "selected_points": entries[selected_index]["points"], "combined_unsmoothed_peak": _peak(weighted, experiment)}


def _combined_map(labels: list[str], reference: dict[str, dict[str, object]]) -> np.ndarray:
    maps = [np.asarray(reference[label]["map"], dtype=float) / float(reference[label]["map_rms"]) for label in labels]
    weights = np.asarray([int(label.split("x")[0]) for label in labels], dtype=float)
    return _combine_arrays(maps, weights)


def _combine_arrays(maps: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    numerator = np.nansum([weight * value for weight, value in zip(weights, maps, strict=True)], axis=0)
    denominator = np.nansum([weight * np.isfinite(value) for weight, value in zip(weights, maps, strict=True)], axis=0)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0.0)


def _bootstrap_map_for_labels(labels: list[str], reference: dict[str, dict[str, object]], experiment: ExperimentData, rng: np.random.Generator) -> np.ndarray:
    indices = rng.integers(0, np.asarray(reference[labels[0]]["raw_gap"]).shape[0], size=np.asarray(reference[labels[0]]["raw_gap"]).shape[0])
    maps: list[np.ndarray] = []
    for label in labels:
        raw_gap = np.asarray(reference[label]["raw_gap"], dtype=float)[indices]
        temporal_weights = np.asarray(reference[label]["temporal_weights"], dtype=float)[indices]
        valid = np.asarray(reference[label]["valid"], dtype=bool)
        weighted_squared = raw_gap**2 * temporal_weights[:, None, None]
        counts = np.sum(np.isfinite(weighted_squared), axis=0)
        values = np.sqrt(np.divide(np.nansum(weighted_squared, axis=0), counts, out=np.full(valid.shape, np.nan), where=counts > 0))
        values[~valid] = np.nan
        maps.append(values / float(reference[label]["map_rms"]))
    weights = np.asarray([int(label.split("x")[0]) for label in labels], dtype=float)
    return _combine_arrays(maps, weights)


def _bootstrap_peaks(raw_gap: np.ndarray, temporal_weights: np.ndarray, valid: np.ndarray, experiment: ExperimentData, rng: np.random.Generator, samples: int) -> np.ndarray:
    peaks = np.empty((samples, 2), dtype=float)
    for index in range(samples):
        selected = rng.integers(0, raw_gap.shape[0], size=raw_gap.shape[0])
        values = raw_gap[selected]
        weighted_squared = values**2 * temporal_weights[selected, None, None]
        counts = np.sum(np.isfinite(weighted_squared), axis=0)
        rms = np.sqrt(np.divide(np.nansum(weighted_squared, axis=0), counts, out=np.full(valid.shape, np.nan), where=counts > 0))
        rms[~valid] = np.nan
        peak = _peak(rms, experiment)
        peaks[index] = (peak["x_mm"], peak["y_mm"])
    return peaks


def _smooth_nan(values: np.ndarray, points: int) -> np.ndarray:
    finite = np.isfinite(values)
    total = uniform_filter(np.where(finite, values, 0.0), size=points)
    support = uniform_filter(finite.astype(float), size=points)
    return np.divide(total, support, out=np.full_like(values, np.nan), where=support > 0.0)


def _peak(values: np.ndarray, experiment: ExperimentData) -> dict[str, float]:
    row, col = np.unravel_index(np.nanargmax(values), values.shape)
    return {"x_mm": float(experiment.specimen_geometry.x[row, col]), "y_mm": float(experiment.specimen_geometry.y[row, col]), "value": float(values[row, col])}


def _plot_scale_diagnostics(diagnostics: dict[str, dict[str, object]], lref: float, path: Path) -> None:
    entries = list(diagnostics.items())
    fractions = [float(value["fraction_of_reference_length"]) for _, value in entries]
    fill = [float(value["fill_p05"]) for _, value in entries]
    spread = [float(value["temporal_peak_spread_mm"]) / lref for _, value in entries]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    axes[0].plot(fractions, fill, marker="o")
    axes[0].axhline(0.5, color="0.4", linestyle="--")
    axes[0].set(xlabel="Window / smaller ROI dimension", ylabel="5th-percentile fill", title="Support quality")
    axes[1].plot(fractions, spread, marker="o")
    axes[1].axhline(0.10, color="0.4", linestyle="--")
    axes[1].set(xlabel="Window / smaller ROI dimension", ylabel="95% peak spread / Lref", title="Temporal bootstrap")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_peak_locations(diagnostics: dict[str, dict[str, object]], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.3, 4.5), layout="constrained")
    for label, entry in diagnostics.items():
        peak = entry["peak"]
        axis.scatter(peak["x_mm"], peak["y_mm"], label=label)
    axis.set(xlabel="x [mm]", ylabel="y [mm]", title="Nominal-normalized EGI peaks by physical scale")
    axis.legend(title="Window [points]")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_noise_stability(diagnostics: dict[str, dict[str, object]], lref: float, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.8, 4.5), layout="constrained")
    for label, entry in diagnostics.items():
        ratios = [float(key) for key in entry["noise"]]
        spreads = [float(value["peak_spread_mm"]) / lref for value in entry["noise"].values()]
        axis.plot(ratios, spreads, marker="o", label=label)
    axis.axhline(0.10, color="0.4", linestyle="--", label="selection limit")
    axis.set(xlabel="Gaussian strain noise / loaded strain RMS", ylabel="95% peak spread / Lref", title="Noise robustness")
    axis.legend(title="Window [points]")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_smoothing(smoothing: dict[str, object], lref: float, path: Path) -> None:
    entries = smoothing["entries"]
    fig, axis = plt.subplots(figsize=(6.0, 4.0), layout="constrained")
    axis.plot([entry["points"] for entry in entries], [entry["peak_spread_mm"] / lref for entry in entries], marker="o")
    axis.axvline(smoothing["selected_points"], color="tab:green", linestyle="--", label="selected")
    axis.set(xlabel="Uniform smoothing width [points]", ylabel="Peak spread / Lref", title="Peak-smoothing stability")
    axis.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_selected_maps(selected: dict[str, object], reference: dict[str, dict[str, object]], points: int, path: Path) -> None:
    labels = list(selected["windows"])
    values = _combined_map(labels, reference)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), layout="constrained")
    for axis, image_values, title in zip(axes, (values, _smooth_nan(values, points)), ("Combined EGI", f"Smoothed ({points} points)"), strict=True):
        image = axis.imshow(image_values, origin="lower", cmap="viridis")
        axis.set(title=title, xticks=[], yticks=[])
        fig.colorbar(image, ax=axis)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_report(summary: dict[str, object], path: Path) -> None:
    selection = summary["selection"]
    smoothing = summary["smoothing"]
    lines = [
        "# Notched-EBW EGI-window selection",
        "",
        "Metric-only Bucket 1C–1F result. Nominal point-count normalization was retained; no known parameter map or identification result was used to select this configuration.",
        "",
        f"- Reference length: {summary['grid']['reference_length_mm']:.3f} mm.",
        f"- Selected windows: {', '.join(selection['windows'])} points.",
        f"- Selected peak smoothing: {smoothing['selected_points']} points.",
        "- Next action: run the 15,500-evaluation matched baseline and selected configuration (Bucket 1G).",
        "",
        "See `figures/` and `summary.json` for the selection evidence and limitations.",
        "",
    ]
    path.write_text("\n".join(lines))


def _grid_spacing(experiment: ExperimentData) -> tuple[float, float]:
    return (float(np.nanmedian(np.diff(experiment.specimen_geometry.x, axis=1))), float(np.nanmedian(np.diff(experiment.specimen_geometry.y, axis=0))))


def _span(values: np.ndarray) -> float:
    return float(np.nanmax(values) - np.nanmin(values))


def _nan_rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(np.asarray(values, dtype=float) ** 2)))


def _percentile(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, percentile))


def _peak_spread(peaks: np.ndarray) -> float:
    centre = np.median(peaks, axis=0)
    return float(np.percentile(np.hypot(peaks[:, 0] - centre[0], peaks[:, 1] - centre[1]), 95.0))


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    return float(np.corrcoef(first[valid], second[valid])[0, 1])


def _odd_round(value: float) -> int:
    return max(1, 2 * int(np.floor(value / 2.0)) + 1)


if __name__ == "__main__":
    main()
