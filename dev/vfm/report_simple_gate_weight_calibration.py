"""Create an offline, truth-free calibration report for simple sensitivity gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


START_QUANTILES = (0.0, 0.10, 0.25, 0.50)
FULL_QUANTILES = (0.50, 0.75, 0.90, 0.95)
RECOMMENDED = (0.0, 0.90)


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = _load_arrays(args.artifact_dir)
    y_activity = arrays["payload__parameter_activity_scaled__yield_strength"]
    h_activity = arrays["payload__parameter_activity_scaled__hardening_modulus"]
    combined = np.fmax(y_activity, h_activity)
    positive = combined[np.isfinite(combined) & (combined > 0.0)]
    start, full = np.quantile(positive, RECOMMENDED)
    recommended_weights = _smooth_gate(combined, float(start), float(full))
    recommended_weights[~np.isfinite(combined)] = np.nan
    current_weights = arrays.get("payload__weights")
    rows = _sweep(combined, y_activity, h_activity)
    recommendation = {
        "selection_rule": "fixed positive-activity quantiles; no known-map input",
        "gate_start_quantile": RECOMMENDED[0],
        "gate_full_quantile": RECOMMENDED[1],
        "resolved_gate_start": float(start),
        "resolved_gate_full": float(full),
        "metrics": _metrics(recommended_weights, y_activity, h_activity),
        "candidate_rows": rows,
        "guard_weight_policy": {
            "status": "retain declared weights for next diagnostic run",
            "force_weight": 0.15,
            "broad_guard_weight": 0.10,
            "rule": "tune later from noise-normalised controlled perturbation discrimination, never yield-map truth",
        },
    }
    with PdfPages(args.output, metadata={"Title": "Simple VFM gate and weight calibration"}) as pdf:
        _overview_page(pdf, recommendation)
        _sweep_page(pdf, rows)
        _map_page(pdf, y_activity, h_activity, current_weights, recommended_weights)
        _temporal_page(pdf, y_activity, h_activity, recommended_weights)
        _history_page(pdf, args.historical_output_root)
    args.output.with_suffix(".json").write_text(
        json.dumps(recommendation, indent=2), encoding="utf-8"
    )
    print(json.dumps({"pdf": str(args.output), "json": str(args.output.with_suffix('.json'))}, indent=2))


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--historical-output-root", type=Path, default=Path("dev/vfm/output"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_arrays(root: Path) -> dict[str, np.ndarray]:
    paths = sorted(root.glob("simple_sensitivity_gate_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No simple sensitivity-gate artefact in {root}.")
    with np.load(paths[0]) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def _smooth_gate(activity, start, full):
    position = np.clip((activity - start) / (full - start), 0.0, 1.0)
    return position**2 * (3.0 - 2.0 * position)


def _metrics(weights, y_activity, h_activity):
    valid = np.isfinite(weights)
    positive = valid & (weights > 0.0)
    transition = positive & (weights < 1.0)
    finite_weights = np.where(valid, weights, 0.0)
    total = float(np.sum(finite_weights))
    ess = total**2 / max(float(np.sum(finite_weights**2)), np.finfo(float).eps)
    def capture(activity):
        values = np.where(np.isfinite(activity), activity, 0.0)
        return float(np.sum(finite_weights * values) / max(np.sum(values), np.finfo(float).eps))
    return {
        "positive_fraction_of_valid": float(np.sum(positive) / np.sum(valid)),
        "transition_fraction_of_positive": float(np.sum(transition) / max(np.sum(positive), 1)),
        "effective_sample_fraction_of_valid": float(ess / np.sum(valid)),
        "yield_activity_capture": capture(y_activity),
        "hardening_activity_capture": capture(h_activity),
    }


def _sweep(combined, y_activity, h_activity):
    positive = combined[np.isfinite(combined) & (combined > 0.0)]
    rows = []
    for start_q in START_QUANTILES:
        for full_q in FULL_QUANTILES:
            if full_q <= start_q:
                continue
            start, full = np.quantile(positive, (start_q, full_q))
            if full <= start:
                continue
            weights = _smooth_gate(combined, float(start), float(full))
            weights[~np.isfinite(combined)] = np.nan
            rows.append({
                "start_quantile": start_q,
                "full_quantile": full_q,
                "resolved_start": float(start),
                "resolved_full": float(full),
                **_metrics(weights, y_activity, h_activity),
            })
    return rows


def _overview_page(pdf, recommendation):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(.06, .92, "Truth-free calibration of sensitivity gates and objective weights", fontsize=20, weight="bold")
    m = recommendation["metrics"]
    lines = [
        "Fixed algorithm; tunable, dimensionless diagnostics",
        "",
        "Recommended next diagnostic setting",
        f"  smooth ramp: positive-activity minimum → q90 (resolved {recommendation['resolved_gate_start']:.3f} → {recommendation['resolved_gate_full']:.3f})",
        f"  positive observations: {100*m['positive_fraction_of_valid']:.1f}% of valid space-time points",
        f"  transition observations: {100*m['transition_fraction_of_positive']:.1f}% of retained points",
        f"  effective sample size: {100*m['effective_sample_fraction_of_valid']:.1f}% of valid observations",
        f"  activity capture: yield {100*m['yield_activity_capture']:.1f}%; hardening {100*m['hardening_activity_capture']:.1f}%",
        "",
        "What may be tuned without making the method specimen-specific",
        "  • positive-activity quantiles defining the smooth gate ramp",
        "  • EGI SNR and coverage gates, using propagated noise for each support",
        "  • explicit FRE and broad-EGI guard shares",
        "  • optional gate refresh interval, tested only after the frozen version",
        "",
        "What must not enter tuning",
        "  • the known synthetic yield map, yielded-region RMSE, or the desired BF locations",
        "",
        "Guard weights remain 0.15 FRE and 0.10 full broad EGI for the next diagnostic run. Change them only after controlled, noise-normalised perturbation tests show domination or loss of closure.",
    ]
    wrapped = []
    import textwrap
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=112, subsequent_indent="    ") or [""])
    fig.text(.075, .84, "\n".join(wrapped), va="top", fontsize=10.8, linespacing=1.36)
    pdf.savefig(fig); plt.close(fig)


def _sweep_page(pdf, rows):
    metrics = (
        ("transition_fraction_of_positive", "Transition share"),
        ("effective_sample_fraction_of_valid", "Effective sample fraction"),
        ("yield_activity_capture", "Yield activity capture"),
        ("hardening_activity_capture", "Hardening activity capture"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
    for axis, (key, title) in zip(axes.ravel(), metrics, strict=True):
        grid = np.full((len(START_QUANTILES), len(FULL_QUANTILES)), np.nan)
        for row in rows:
            grid[START_QUANTILES.index(row["start_quantile"]), FULL_QUANTILES.index(row["full_quantile"])] = row[key]
        image = axis.imshow(grid, vmin=0, vmax=1, cmap="viridis", origin="lower")
        axis.set_xticks(range(len(FULL_QUANTILES)), [f"q{int(100*q)}" for q in FULL_QUANTILES])
        axis.set_yticks(range(len(START_QUANTILES)), [f"q{int(100*q)}" for q in START_QUANTILES])
        axis.set(xlabel="Gate fully active", ylabel="Gate begins", title=title)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isfinite(grid[i, j]): axis.text(j, i, f"{100*grid[i,j]:.0f}%", ha="center", va="center", color="white" if grid[i,j] < .55 else "black")
        fig.colorbar(image, ax=axis, shrink=.78)
    fig.suptitle("Offline gate sweep: no known material map used", fontsize=17)
    pdf.savefig(fig); plt.close(fig)


def _rms_time(values):
    return np.sqrt(_mean_time(values**2))


def _mean_time(values):
    resolved = np.asarray(values, dtype=float)
    valid = np.isfinite(resolved)
    return np.divide(
        np.nansum(resolved, axis=0), np.sum(valid, axis=0),
        out=np.full(resolved.shape[1:], np.nan), where=np.sum(valid, axis=0) > 0,
    )


def _map_page(pdf, y_activity, h_activity, current, recommended):
    panels = [("Yield sensitivity", _rms_time(y_activity)), ("Hardening sensitivity", _rms_time(h_activity))]
    if current is not None: panels.append(("Previous gate", _mean_time(current)))
    panels.append(("Recommended minimum→q90 gate", _mean_time(recommended)))
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
    for axis, (title, values) in zip(axes.ravel(), panels, strict=False):
        image = axis.imshow(values, origin="lower", cmap="magma", aspect="auto")
        axis.set_title(title); fig.colorbar(image, ax=axis, shrink=.75)
    for axis in axes.ravel()[len(panels):]: axis.axis("off")
    fig.suptitle("Sensitivity activity and frozen observation gates", fontsize=17)
    pdf.savefig(fig); plt.close(fig)


def _temporal_page(pdf, y_activity, h_activity, weights):
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27), constrained_layout=True)
    for values, label in ((y_activity, "yield"), (h_activity, "hardening")):
        axes[0].plot(np.sqrt(np.nanmean(values**2, axis=(1, 2))), marker="o", label=label)
    axes[0].set(title="Frame-wise sensitivity activity", xlabel="Frame", ylabel="RMS scaled activity"); axes[0].legend(); axes[0].grid(alpha=.3)
    valid = np.isfinite(weights)
    axes[1].plot(np.nanmean(weights, axis=(1, 2)), marker="o", label="mean gate weight")
    axes[1].plot(np.sum((weights > 0) & valid, axis=(1, 2)) / np.maximum(np.sum(valid, axis=(1, 2)), 1), marker="s", label="positive fraction")
    axes[1].set(title="Temporal observation retention", xlabel="Frame", ylabel="Fraction / mean weight"); axes[1].legend(); axes[1].grid(alpha=.3)
    pdf.savefig(fig); plt.close(fig)


def _history_page(pdf, root):
    weighting_path = root / "SPATIAL_WEIGHTING_COMPARISON_20260827.json"
    windows_path = root / "egi_window_selection_20260827/summary.json"
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(.06, .92, "How previous investigations constrain—but do not fit—the method", fontsize=19, weight="bold")
    if not (weighting_path.is_file() and windows_path.is_file()):
        lines = [
            "Historical comparison unavailable in this checkout.",
            "",
            "The gate and weight calibration above is complete: it uses only the current run's frozen sensitivity artefact and does not require known-map truth.",
            "",
            "Optional historical artefacts can be supplied later to add cross-investigation context. Their absence must not prevent a completed identification from producing its calibration report.",
        ]
        fig.text(.075, .83, "\n".join(lines), va="top", fontsize=11.5, linespacing=1.48, wrap=True)
        pdf.savefig(fig); plt.close(fig)
        return

    weighting = json.loads(weighting_path.read_text())
    windows = json.loads(windows_path.read_text())
    change = weighting["weighted_relative_change_percent"]
    lines = [
        "Reusable evidence",
        f"  • Earlier sensitivity weighting changed yielded RMSE by {change['yielded_rmse_mpa']:+.1f}% and high-plastic RMSE by {change['high_plastic_rmse_mpa']:+.1f}%, but increased runtime by {change['total_runtime_seconds']:+.1f}%.",
        "  • This supports informative-region weighting, but not the old floor/normalisation or any truth-tuned threshold.",
        f"  • The earlier metric-only window audit selected {windows['selection']['local_window']} as its stable local scale and found that larger scales lose coverage and add coarse information.",
        "  • The latest direct-SNR sweep preserves that physical progression while avoiding a redundancy/Fisher optimisation.",
        "  • Prior objective-discrimination work showed that FRE and EGI carry complementary information and can compensate for each other; guards must therefore remain explicit and separately reported.",
        "",
        "General validation sequence",
        "  1. Propagate realistic DIC noise through every normalised-EGI support.",
        "  2. Freeze support selection and a quantile gate without consulting truth.",
        "  3. Check observation continuity, temporal coverage, parameter capture and effective component contributions.",
        "  4. Run BF0–BF1 and inspect physical maps; use truth only as a held-out synthetic evaluation.",
        "  5. Replicate across noise realisations/specimen geometries before changing defaults.",
    ]
    fig.text(.075, .83, "\n".join(lines), va="top", fontsize=11.5, linespacing=1.48, wrap=True)
    pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
