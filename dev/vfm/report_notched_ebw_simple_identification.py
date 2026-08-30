"""Create a concise PDF for a minimalist sensitivity-gated BF identification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from pyvale.vfm import ExperimentData, load_identification_result
from pyvale.vfm.postprocessing import load_constitutive_law_from_result, load_known_parameter_maps
import report_notched_ebw_data_driven_identification as common


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    experiment = ExperimentData.load_from_file(args.input / "experiment_data.yaml")
    known = load_known_parameter_maps(args.input / "known_parameter_maps.npz")
    result = load_identification_result(args.run / "identification_result.yaml")
    law = load_constitutive_law_from_result(result)
    mask = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(
        experiment.specimen_geometry.x, experiment.specimen_geometry.y
    )
    truth = np.asarray(known["yield_strength"], dtype=float)
    states = common._states(result, experiment, known)
    supports = common._selected_supports(result)
    egi_maps = common._egi_maps(states, supports, experiment, law)
    gate = _load_gate(args.run / "diagnostic_artifacts")
    summary = _summary(result, states, truth, mask, supports, gate)
    with PdfPages(args.output, metadata={"Title": "Simple sensitivity-gated identification"}) as pdf:
        _summary_page(pdf, summary)
        _algorithm_page(pdf, summary)
        common._yield_map_page(pdf, states, truth, mask, experiment)
        common._egi_page(pdf, states, supports, egi_maps, mask, experiment)
        _gate_page(pdf, gate, experiment)
        _components_page(pdf, result)
        common._basis_page(pdf, states, egi_maps, supports, mask, experiment)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"pdf": str(args.output), "json": str(args.output.with_suffix('.json'))}, indent=2))


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_gate(root):
    paths = sorted(root.glob("simple_sensitivity_gate_*.npz"))
    if not paths: raise FileNotFoundError(f"No simple sensitivity gate in {root}")
    with np.load(paths[0]) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def _summary(result, states, truth, mask, supports, gate):
    rows = []
    for state in states:
        values = np.asarray(state["maps"]["yield_strength"])
        rows.append({
            "state": state["label"],
            "rmse_mpa": float(np.sqrt(np.mean((values[mask] - truth[mask])**2))),
            "range_mpa": [float(np.min(values[mask])), float(np.max(values[mask]))],
        })
    solve = result.history.phases[1].solve_results[-1]
    diagnostics = solve.final_objective.get("objective_diagnostics", {})
    costs = solve.final_objective.get("components", {})
    training_weight, force_weight, broad_weight = .75, .15, .10
    return {
        "interpretation": "engineering diagnostic; known truth used only for report evaluation",
        "selected_supports": {role: list(window) for role, window in supports},
        "states": rows,
        "gate": {
            "positive_fraction": diagnostics.get("gate_positive_fraction"),
            "transition_fraction": diagnostics.get("gate_transition_fraction_of_positive"),
            "effective_sample_fraction": diagnostics.get("gate_effective_sample_fraction"),
            "parameter_activity_capture": diagnostics.get("parameter_activity_capture"),
            "mean_positive": diagnostics.get("gate_mean_positive"),
            "resolved_start": diagnostics.get("gate_start"),
            "resolved_full": diagnostics.get("gate_full"),
            "start_quantile": diagnostics.get("gate_start_quantile"),
            "full_quantile": diagnostics.get("gate_full_quantile"),
        },
        "objective_components": costs,
        "effective_contributions": {
            "informative_egi": training_weight * costs.get("informative_egi_cost", np.nan),
            "fre_guard": force_weight * costs.get("force_guard_cost", np.nan),
            "broad_egi_guard": broad_weight * costs.get("broad_guard_cost", np.nan),
        },
    }


def _summary_page(pdf, summary):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(.06, .92, "Minimal sensitivity-gated EGI identification", fontsize=21, weight="bold")
    supports = summary["selected_supports"]; gate = summary["gate"]
    gate_label = (
        f"positive-activity q{100*gate['start_quantile']:.0f}→q{100*gate['full_quantile']:.0f}"
        if gate["start_quantile"] is not None
        else "fixed absolute activity thresholds"
    )
    lines = [
        "Status: engineering diagnostic. The known synthetic map appears only in evaluation pages, never in support/gate/objective tuning.", "",
        "Selected EGI supports: " + ", ".join(f"{role} {v[0]}×{v[1]}" for role, v in supports.items()),
        f"Frozen gate: {gate_label}; resolved {gate['resolved_start']:.3f}→{gate['resolved_full']:.3f}; retains {100*gate['positive_fraction']:.1f}% of valid observations, with {100*(gate['transition_fraction'] or 0):.1f}% in transition.", "",
        "Yield-map evaluation", *[
            f"  {row['state']}: RMSE {row['rmse_mpa']:.2f} MPa; range {row['range_mpa'][0]:.1f}–{row['range_mpa'][1]:.1f} MPa"
            for row in summary["states"]
        ], "",
        "Effective terminal objective contributions", *[
            f"  {name}: {value:.3g}" for name, value in summary["effective_contributions"].items()
        ], "",
        "Review gates: sensitivity should cover physically active frames/regions without becoming binary; BF geometry should remain specimen-scale; FRE and broad closure must not be sacrificed for local EGI reduction.",
    ]
    import textwrap
    wrapped=[]
    for line in lines: wrapped.extend(textwrap.wrap(line, width=112, subsequent_indent="    ") or [""])
    fig.text(.075, .84, "\n".join(wrapped), va="top", fontsize=10.8, linespacing=1.4)
    pdf.savefig(fig); plt.close(fig)


def _algorithm_page(pdf, summary):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(.06, .92, "Algorithm used", fontsize=21, weight="bold")
    lines = [
        "1. Homogeneous phase: identify global yield strength and hardening with SBVF + least squares.",
        "2. Evaluate every odd EGI window from 3 points to the half-bounding-box limit on the accepted homogeneous stress.",
        "3. Divide each normalised EGI field by its support-specific noise; select smallest resolved, logarithmic middle and largest covered support.",
        "4. Reconstruct stress twice: one global yield perturbation and one global hardening perturbation.",
        "5. Robustly scale their pointwise space-time magnitudes, combine by maximum, and freeze a smooth positive-activity quantile gate.",
        "6. Optimise BF parameters with ordinary noise-normalised RMS terms:",
        "       0.75 × mean(fine, middle, broad informative EGI)",
        "     + 0.15 × unmasked FRE guard",
        "     + 0.10 × unmasked full broad-EGI guard.",
        "7. No local probes, optimiser-coordinate sensitivity matrix, SVD, Fisher matrix or projected residual.",
    ]
    y=.82
    for line in lines:
        fig.text(.085, y, line, fontsize=12, va="top"); y -= .072 if line[:1].isdigit() else .05
    pdf.savefig(fig); plt.close(fig)


def _gate_page(pdf, gate, experiment):
    y = gate["payload__parameter_activity_scaled__yield_strength"]
    h = gate["payload__parameter_activity_scaled__hardening_modulus"]
    w = gate["payload__weights"]
    panels = [
        ("Yield sensitivity temporal RMS", np.sqrt(_mean_time(y*y))),
        ("Hardening sensitivity temporal RMS", np.sqrt(_mean_time(h*h))),
        ("Mean frozen gate weight", _mean_time(w)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
    extent = common._extent(experiment)
    for axis, (title, values) in zip(axes.ravel(), panels, strict=False):
        image=axis.imshow(values, origin="lower", extent=extent, aspect="equal", cmap="magma")
        axis.set_title(title); axis.set(xlabel="x [mm]", ylabel="y [mm]"); fig.colorbar(image, ax=axis, shrink=.75)
    axes[1,1].plot(np.sqrt(np.nanmean(y*y,axis=(1,2))), marker="o", label="yield")
    axes[1,1].plot(np.sqrt(np.nanmean(h*h,axis=(1,2))), marker="o", label="hardening")
    axes[1,1].plot(np.nanmean(w,axis=(1,2)), marker="s", label="mean gate")
    axes[1,1].set(title="Temporal activity and gate", xlabel="Frame"); axes[1,1].legend(); axes[1,1].grid(alpha=.3)
    fig.suptitle("Two-perturbation sensitivity gate used by the solve", fontsize=17)
    pdf.savefig(fig); plt.close(fig)


def _mean_time(values):
    resolved = np.asarray(values, dtype=float)
    valid = np.isfinite(resolved)
    count = np.sum(valid, axis=0)
    return np.divide(
        np.nansum(resolved, axis=0), count,
        out=np.full(resolved.shape[1:], np.nan), where=count > 0,
    )


def _components_page(pdf, result):
    solve=result.history.phases[1].solve_results[-1]
    history=solve.final_objective.get("history", [])
    components=solve.final_objective.get("components", {})
    fig,axes=plt.subplots(1,2,figsize=(11.69,8.27),constrained_layout=True)
    if history:
        axes[0].plot([row["iteration"] for row in history],[row["cost"] for row in history],marker="o")
    axes[0].set(title="Optimiser trajectory",xlabel="Iteration",ylabel="Total cost"); axes[0].grid(alpha=.3)
    names=["informative_egi_cost","force_guard_cost","broad_guard_cost"]
    raw=np.array([components.get(name,np.nan) for name in names]); coeff=np.array([.75,.15,.10])
    x=np.arange(3); axes[1].bar(x-.18,raw,.36,label="raw"); axes[1].bar(x+.18,raw*coeff,.36,label="effective")
    axes[1].set_xticks(x,["informative EGI","FRE guard","broad guard"]); axes[1].set(title="Terminal component balance",ylabel="Cost contribution"); axes[1].legend(); axes[1].grid(axis="y",alpha=.3)
    pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
