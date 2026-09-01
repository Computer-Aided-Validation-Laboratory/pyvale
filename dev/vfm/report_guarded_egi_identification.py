"""Generate a concise guarded-EGI campaign report from one completed bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
import numpy as np

from pyvale.vfm import ExperimentData, VfmRegionOfInterest, load_identification_result
from pyvale.vfm.postprocessing import evaluate_snapshot_parameter_maps, load_known_parameter_maps


PRELIMINARY_LABEL = (
    "Preliminary WDBN1 experimental identification using the current "
    "guarded EGI-primary candidate algorithm."
)


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output.parent / f"{args.output.stem}_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    experiment = ExperimentData.load_from_file(args.input / "experiment_data.yaml")
    if args.fre_region_of_interest is not None:
        experiment.specimen_geometry.force_reconstruction_region_of_interest = (
            VfmRegionOfInterest.from_yaml(args.fre_region_of_interest)
        )
    result = load_identification_result(args.run / "identification_result.yaml")
    known = load_known_parameter_maps(None, args.input)
    truth = None if known is None else known.get("yield_strength")
    mask = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(
        experiment.specimen_geometry.x, experiment.specimen_geometry.y
    )
    states = _states(result, experiment, mask)
    preparations = _artifacts(args.run, "guarded_egi_preparation")
    solve_summaries = _artifacts(args.run, "guarded_egi_solve_summary")
    summary = _summary(args, result, states, truth, mask, preparations, solve_summaries)

    with PdfPages(args.output, metadata={"Title": args.title}) as pdf:
        _overview_page(pdf, args, result, summary)
        _montage_page(pdf, experiment, states, truth, mask, figure_dir)
        _final_map_page(pdf, experiment, states[-1], truth, mask, figure_dir)
        _progress_page(pdf, states, truth, mask, preparations, figure_dir)
        _fre_page(pdf, solve_summaries, figure_dir)
        _guard_page(pdf, preparations, solve_summaries)
        _basis_page(pdf, experiment, states, mask, figure_dir)
        if args.fre_region_of_interest is not None:
            _roi_page(pdf, experiment, solve_summaries, mask, figure_dir)

    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"pdf": str(args.output), "figures": str(figure_dir)}, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Guarded EGI-primary identification")
    parser.add_argument("--fre-region-of-interest", type=Path)
    parser.add_argument("--experimental", action="store_true")
    return parser.parse_args()


def _states(result, experiment, mask):
    entries = [("Phase0", result.history.phases[0].final_snapshot, None)]
    for solve in result.history.phases[1].solve_results:
        if solve.final_snapshot is not None:
            entries.append((f"BF{_basis_count(solve.final_snapshot)}", solve.final_snapshot, solve))
    states = []
    for label, snapshot, solve in entries:
        if snapshot is None:
            continue
        maps = evaluate_snapshot_parameter_maps(snapshot, experiment)
        yield_map = np.asarray(maps["yield_strength"], dtype=np.float64)
        yield_map = np.where(mask, yield_map, np.nan)
        hardening = maps.get("hardening_modulus")
        states.append({
            "label": label,
            "yield_strength": yield_map,
            "hardening_mean": None if hardening is None else float(np.nanmean(np.where(mask, hardening, np.nan))),
            "solve": solve,
            "kernels": _kernels(snapshot),
        })
    return states


def _basis_count(snapshot) -> int:
    return len(_kernels(snapshot))


def _kernels(snapshot):
    for item in snapshot.spatial_parameterisations.get("yield_strength", []):
        if item.summary.get("kind") == "basis_functions":
            return item.summary.get("kernels", [])
    return []


def _artifacts(run: Path, kind: str):
    records = []
    for metadata_path in sorted((run / "diagnostic_artifacts").glob(f"{kind}_*.json")):
        arrays = {}
        arrays_path = metadata_path.with_suffix(".npz")
        if arrays_path.is_file():
            with np.load(arrays_path) as loaded:
                arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
        records.append(_decode(json.loads(metadata_path.read_text()), arrays))
    return records


def _decode(value, arrays):
    if isinstance(value, dict) and "array_key" in value:
        return arrays[value["array_key"]]
    if isinstance(value, dict):
        return {key: _decode(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item, arrays) for item in value]
    return value


def _summary(args, result, states, truth, mask, preparations, solve_summaries):
    rows = []
    for state in states:
        values = state["yield_strength"]
        row = {
            "stage": state["label"],
            "yield_mean_mpa": float(np.nanmean(values)),
            "yield_min_mpa": float(np.nanmin(values)),
            "yield_max_mpa": float(np.nanmax(values)),
            "hardening_mean_mpa": state["hardening_mean"],
        }
        if truth is not None:
            error = values[mask] - np.asarray(truth)[mask]
            reference = np.asarray(truth)[mask]
            row.update({
                "truth_rmse_mpa": float(np.sqrt(np.mean(error**2))),
                "truth_mape_percent": float(100.0 * np.mean(np.abs(error / reference))),
                "truth_p95_absolute_error_mpa": float(np.percentile(np.abs(error), 95.0)),
            })
        rows.append(row)
    aggregate = _aggregate_candidates(solve_summaries)
    return {
        "label": PRELIMINARY_LABEL if args.experimental else "Synthetic guarded EGI-primary identification; truth used only in this post-run report.",
        "run_started_at": result.metadata.run.started_at,
        "run_finished_at": result.metadata.run.finished_at,
        "runtime_seconds": result.metadata.run.runtime_seconds,
        "input": str(args.input),
        "run": str(args.run),
        "fre_domain_correction": result.metadata.input.force_reconstruction_domain_correction,
        "states": rows,
        "candidate_short_circuit": aggregate,
        "guard_stages": len(preparations),
    }


def _aggregate_candidates(summaries):
    keys = ("total_candidate_evaluations", "rejected_at_fre", "rejected_at_broad", "reaching_fine_egi")
    totals = {key: int(sum(int(row.get(key, 0)) for row in summaries)) for key in keys}
    count = totals["total_candidate_evaluations"]
    for key in keys[1:]:
        totals[f"{key}_fraction"] = 0.0 if count == 0 else totals[key] / count
    for key in ("total_stress_time_seconds", "total_fre_time_seconds", "total_broad_egi_time_seconds", "total_fine_egi_time_seconds"):
        totals[key] = float(sum(float(row.get(key, 0.0)) for row in summaries))
    return totals


def _overview_page(pdf, args, result, summary):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle(args.title, fontsize=20, fontweight="bold")
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.80]); ax.axis("off")
    short = summary["candidate_short_circuit"]
    lines = [
        summary["label"], "",
        "Frozen algorithm", "J = mean(fine gated/noise scale, broad gated/noise scale); no middle EGI.",
        "Hard guards: FRE then unmasked broad EGI <= 1.10 x max(parent, measurement-noise floor).",
        "SPD Gaussian yield field; sensitivity-correction placement; global refit; fixed BF1-BF7 trajectory.", "",
        f"Started: {summary['run_started_at']}    Finished: {summary['run_finished_at']}",
        f"Runtime: {summary['runtime_seconds']} s    Accepted states: {len(summary['states'])}",
        f"FRE physical-domain correction: {summary['fre_domain_correction']}", "",
        "Measured short-circuit behaviour",
        f"Candidates: {short['total_candidate_evaluations']}; rejected at FRE: {short['rejected_at_fre_fraction']:.1%}; "
        f"rejected at broad: {short['rejected_at_broad_fraction']:.1%}; reached fine: {short['reaching_fine_egi_fraction']:.1%}.", "",
        "State summary",
    ]
    for row in summary["states"]:
        truth_text = "" if "truth_rmse_mpa" not in row else f", truth RMSE {row['truth_rmse_mpa']:.1f} MPa, MAPE {row['truth_mape_percent']:.1f}%"
        lines.append(f"{row['stage']}: yield {row['yield_mean_mpa']:.1f} MPa mean [{row['yield_min_mpa']:.1f}, {row['yield_max_mpa']:.1f}]{truth_text}")
    ax.text(0, 1, "\n".join(lines), va="top", fontsize=10.5, linespacing=1.35)
    pdf.savefig(fig); plt.close(fig)


def _montage_page(pdf, experiment, states, truth, mask, figure_dir):
    fig, axes = plt.subplots(2, 4, figsize=(11.69, 8.27), constrained_layout=True)
    values = np.concatenate([state["yield_strength"][mask] for state in states])
    vmin, vmax = np.percentile(values, [1, 99])
    image = None
    for ax, state in zip(axes.flat, states, strict=False):
        image = _map(ax, experiment, state["yield_strength"], state["label"], vmin, vmax, "viridis")
    for ax in axes.flat[len(states):]: ax.axis("off")
    fig.suptitle("Phase0 to BF7 identified yield-strength progression")
    if image is not None: fig.colorbar(image, ax=axes, label="Yield strength [MPa]", shrink=0.8)
    _save(pdf, fig, figure_dir / "phase0_to_bf7_yield_montage.png")


def _final_map_page(pdf, experiment, state, truth, mask, figure_dir):
    panels = 2 if truth is not None else 1
    fig, axes = plt.subplots(1, panels, figsize=(11.69, 8.27), constrained_layout=True)
    axes = np.atleast_1d(axes)
    values = state["yield_strength"][mask]
    image = _map(axes[0], experiment, state["yield_strength"], "Final identified yield strength", np.percentile(values, 1), np.percentile(values, 99), "viridis")
    fig.colorbar(image, ax=axes[0], label="Yield strength [MPa]")
    if truth is not None:
        error = 100.0 * (state["yield_strength"] - truth) / truth
        limit = np.percentile(np.abs(error[mask]), 98)
        image = _map(axes[1], experiment, error, "Final error", -limit, limit, "RdBu_r")
        fig.colorbar(image, ax=axes[1], label="Error [%]")
    fig.suptitle("Final accepted identification")
    _save(pdf, fig, figure_dir / "final_yield_map.png")


def _progress_page(pdf, states, truth, mask, preparations, figure_dir):
    labels = [state["label"] for state in states]
    h = [state["hardening_mean"] for state in states]
    primary = [np.nan]
    fine = [np.nan]; broad = [np.nan]
    for state in states[1:]:
        components = {} if state["solve"] is None else state["solve"].final_objective.get("components", {})
        primary.append(components.get("total_cost", np.nan))
        fine.append(components.get("fine_gated_cost", np.nan))
        broad.append(components.get("broad_gated_cost", np.nan))
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)
    axes[0,0].plot(labels, h, "o-"); axes[0,0].set_title("Hardening progression"); axes[0,0].set_ylabel("H [MPa]")
    axes[0,1].plot(labels, primary, "o-", label="primary"); axes[0,1].plot(labels, fine, "o-", label="fine"); axes[0,1].plot(labels, broad, "o-", label="broad"); axes[0,1].set_title("EGI-primary components"); axes[0,1].legend()
    if truth is not None:
        rmse = [np.sqrt(np.mean((state["yield_strength"][mask] - truth[mask])**2)) for state in states]
        axes[1,0].plot(labels, rmse, "o-"); axes[1,0].set_title("Held-out truth RMSE"); axes[1,0].set_ylabel("RMSE [MPa]")
    else:
        axes[1,0].axis("off")
    limits_fre = [item.get("fre_guard", {}).get("limit", np.nan) for item in preparations]
    limits_broad = [item.get("broad_unmasked_guard", {}).get("limit", np.nan) for item in preparations]
    axes[1,1].plot(labels[1:1+len(limits_fre)], limits_fre, "o-", label="FRE limit")
    axes[1,1].plot(labels[1:1+len(limits_broad)], limits_broad, "o-", label="broad limit")
    axes[1,1].set_title("Frozen guard limits"); axes[1,1].legend()
    for ax in axes.flat: ax.tick_params(axis="x", rotation=45)
    _save(pdf, fig, figure_dir / "objective_h_guard_progression.png")


def _fre_page(pdf, summaries, figure_dir):
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27), constrained_layout=True)
    plotted = 0
    for index, row in enumerate(summaries, 1):
        fre = row.get("accepted_fre", {})
        force = np.asarray(fre.get("reconstructed_force_n", []))
        relative = np.asarray(fre.get("relative_fre_percent", []))
        if force.ndim == 2:
            axes[0].plot(np.nanmean(force, axis=0), label=f"BF{index}")
            plotted += 1
        if relative.ndim == 2:
            axes[1].plot(np.sqrt(np.nanmean(relative**2, axis=0)), label=f"BF{index}")
    axes[0].set_title("Accepted-state reconstructed-force profiles (frame mean)"); axes[0].set_ylabel("Force [N]")
    axes[1].set_title("Accepted-state FRE profiles (frame RMS)"); axes[1].set_ylabel("FRE [%]"); axes[1].set_xlabel("Longitudinal slice index")
    if plotted: axes[0].legend(ncol=4); axes[1].legend(ncol=4)
    _save(pdf, fig, figure_dir / "fre_profile_evolution.png")


def _guard_page(pdf, preparations, summaries):
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), constrained_layout=True)
    stages = np.arange(1, len(preparations) + 1)
    for ax, key, title in ((axes[0], "fre_guard", "FRE guard"), (axes[1], "broad_unmasked_guard", "Unmasked broad-EGI guard")):
        for field in ("parent", "noise_floor", "reference", "limit"):
            ax.plot(stages, [row.get(key, {}).get(field, np.nan) for row in preparations], "o-", label=field)
        ax.set_title(title); ax.set_xlabel("BF solve"); ax.legend()
    total = _aggregate_candidates(summaries)
    fig.suptitle(f"Guard references and limits | FRE reject {total['rejected_at_fre_fraction']:.1%}, broad reject {total['rejected_at_broad_fraction']:.1%}, fine reached {total['reaching_fine_egi_fraction']:.1%}")
    pdf.savefig(fig); plt.close(fig)


def _basis_page(pdf, experiment, states, mask, figure_dir):
    state = states[-1]
    fig, ax = plt.subplots(figsize=(11.69, 8.27), constrained_layout=True)
    values = state["yield_strength"][mask]
    image = _map(ax, experiment, state["yield_strength"], "Final yield map and BF geometry", np.percentile(values, 1), np.percentile(values, 99), "viridis")
    for kernel in state["kernels"]:
        width = np.asarray(kernel.get("width", [0.1, 0.1]), dtype=float)
        if width.size == 1: width = np.repeat(width, 2)
        centre = kernel["centre"]
        ellipse = Ellipse(centre, 2*width[0], 2*width[1], angle=np.degrees(float(kernel.get("angle", 0.0))), fill=False, color="white", lw=1.5)
        ax.add_patch(ellipse); ax.plot(*centre, "+", color="white")
    fig.colorbar(image, ax=ax, label="Yield strength [MPa]")
    _save(pdf, fig, figure_dir / "final_bf_overlay.png")


def _roi_page(pdf, experiment, summaries, mask, figure_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), constrained_layout=True)
    axes[0].pcolormesh(experiment.specimen_geometry.x, experiment.specimen_geometry.y, mask.astype(float), shading="auto")
    physical = experiment.specimen_geometry.force_reconstruction_region_of_interest.sample_specimen_mask(experiment.specimen_geometry.x, experiment.specimen_geometry.y)
    axes[0].contour(experiment.specimen_geometry.x, experiment.specimen_geometry.y, physical.astype(float), levels=[0.5], colors="red")
    axes[0].set_title("Measured DIC/EGI ROI (fill) and physical FRE ROI (red)")
    for index, row in enumerate(summaries, 1):
        factors = np.asarray(row.get("accepted_fre", {}).get("force_integration_scale_factors", []))
        if factors.ndim == 1: axes[1].plot(factors, label=f"BF{index}")
    axes[1].set_title("Physical/measured FRE width ratio"); axes[1].set_xlabel("Longitudinal slice index"); axes[1].set_ylabel("Correction multiplier")
    if summaries: axes[1].legend(ncol=2)
    _save(pdf, fig, figure_dir / "fre_roi_and_area_correction.png")


def _map(ax, experiment, values, title, vmin, vmax, cmap):
    image = ax.pcolormesh(experiment.specimen_geometry.x, experiment.specimen_geometry.y, values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title); ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_aspect("equal")
    return image


def _save(pdf, fig, png):
    fig.savefig(png, dpi=300, bbox_inches="tight")
    pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    main()
