"""Create a concise physical diagnostic PDF for one data-driven VFM run."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse, FancyBboxPatch
import numpy as np

from pyvale.vfm import EquilibriumGapMetric, ExperimentData, load_identification_result
from pyvale.vfm.postprocessing import (
    evaluate_snapshot_parameter_maps,
    load_constitutive_law_from_result,
    load_known_parameter_maps,
)


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    experiment = ExperimentData.load_from_file(args.input / "experiment_data.yaml")
    known = load_known_parameter_maps(args.input / "known_parameter_maps.npz")
    if known is None or "yield_strength" not in known:
        raise RuntimeError("Known yield-strength map is required for the synthetic report.")
    result = load_identification_result(args.run / "identification_result.yaml")
    law = load_constitutive_law_from_result(result)
    mask = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(
        experiment.specimen_geometry.x, experiment.specimen_geometry.y
    )
    truth = np.asarray(known["yield_strength"], dtype=np.float64)
    states = _states(result, experiment, known)
    supports = _selected_supports(result)
    egi_maps = _egi_maps(states, supports, experiment, law)
    sensitivity = _load_sensitivity_artifact(args.run / "diagnostic_artifacts")
    preparation = result.history.phases[1].preparation
    summary = _summary(states, truth, mask, supports, sensitivity, preparation, egi_maps)

    with PdfPages(
        args.output,
        metadata={
            "Title": "Data-driven EGI identification smoke diagnostic",
            "Author": "PyVale diagnostic report",
        },
    ) as pdf:
        _summary_page(pdf, summary)
        _algorithm_page(pdf, summary)
        _yield_map_page(pdf, states, truth, mask, experiment)
        _egi_page(pdf, states, supports, egi_maps, mask, experiment)
        _sensitivity_pages(pdf, sensitivity, experiment)
        _basis_page(pdf, states, egi_maps, supports, mask, experiment)

    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"pdf": str(args.output), "summary": str(summary_path)}, indent=2))


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _states(result, experiment, known):
    entries = []
    phase0 = result.history.phases[0].final_snapshot
    if phase0 is None:
        raise RuntimeError("Phase 0 has no final snapshot.")
    entries.append(("Phase 0 homogeneous", phase0, None))
    for solve in result.history.phases[1].solve_results:
        if solve.final_snapshot is None:
            continue
        count = _basis_count(solve.final_snapshot)
        entries.append((f"BF{count} solve", solve.final_snapshot, solve))
    states = []
    for label, snapshot, solve in entries:
        maps = {name: np.asarray(value, dtype=np.float64).copy() for name, value in known.items()}
        maps.update(evaluate_snapshot_parameter_maps(snapshot, experiment))
        states.append({"label": label, "snapshot": snapshot, "solve": solve, "maps": maps})
    return states


def _basis_count(snapshot) -> int:
    for item in snapshot.spatial_parameterisations.get("yield_strength", []):
        if item.summary.get("kind") == "basis_functions":
            return int(item.summary.get("num_kernels", 0))
    return 0


def _selected_supports(result):
    preparation = result.history.phases[1].preparation
    selection = _as_mapping(preparation.get("selection", {}))
    installed = _as_mapping(preparation.get("installed", {}))
    selected = selection.get("selected_supports", {})
    if not selected:
        selected = installed.get("roles", {})
    roles = []
    for role in ("fine", "middle", "broad"):
        support = selected.get(role)
        if support is None:
            raise RuntimeError(f"Run has no selected {role} EGI support diagnostic.")
        roles.append((role, tuple(int(value) for value in support["window_size"])))
    return roles


def _as_mapping(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


def _egi_maps(states, supports, experiment, law):
    metrics = [EquilibriumGapMetric(window_size=window) for _, window in supports]
    for metric in metrics:
        metric.initialise(experiment)
    result = {}
    for state in states:
        stress = law.calculate_stress(experiment.strain, state["maps"])
        maps = {}
        for (role, window), metric in zip(supports, metrics, strict=True):
            fields = metric.evaluate_equilibrium_gap(stress).metric_result.additional_fields or {}
            maps[role] = np.asarray(fields["weighted_temporal_rms"], dtype=np.float64)
        result[state["label"]] = maps
    return result


def _load_sensitivity_artifact(root: Path):
    metadata_paths = sorted(root.glob("solve_sensitivity_*.json"))
    if not metadata_paths:
        raise RuntimeError(f"No solve sensitivity artefact found in {root}.")
    records = []
    for metadata_path in metadata_paths:
        arrays_path = metadata_path.with_suffix(".npz")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(arrays_path) as loaded:
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
        records.append(_decode_arrays(metadata, arrays))
    return records


def _decode_arrays(value, arrays):
    if isinstance(value, dict) and "array_key" in value:
        return arrays[value["array_key"]]
    if isinstance(value, dict):
        return {key: _decode_arrays(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_arrays(item, arrays) for item in value]
    return value


def _summary(states, truth, mask, supports, sensitivities, preparation, egi_maps):
    state_rows = []
    for state in states:
        identified = np.asarray(state["maps"]["yield_strength"], dtype=np.float64)
        error = identified - truth
        state_rows.append({
            "state": state["label"],
            "yield_rmse_mpa": float(np.sqrt(np.mean(error[mask] ** 2))),
            "yield_mean_mpa": float(np.mean(identified[mask])),
            "yield_min_mpa": float(np.min(identified[mask])),
            "yield_max_mpa": float(np.max(identified[mask])),
        })
    evidence = _as_mapping(preparation.get("sweep", {})).get("evidence", [])
    homogeneous_snr = max(
        (item.get("probe_response_to_noise", {}).get("homogeneous_yield", 0.0) for item in evidence),
        default=0.0,
    )
    local_snr = max(
        (
            value
            for item in evidence
            for name, value in item.get("probe_response_to_noise", {}).items()
            if name.startswith("local_yield_")
        ),
        default=0.0,
    )
    initial_rmse = state_rows[0]["yield_rmse_mpa"]
    final_rmse = state_rows[-1]["yield_rmse_mpa"]
    egi_changes = {}
    for role, _ in supports:
        initial = np.asarray(egi_maps[states[0]["label"]][role])
        final = np.asarray(egi_maps[states[-1]["label"]][role])
        initial_mean = float(np.nanmean(initial[mask]))
        final_mean = float(np.nanmean(final[mask]))
        egi_changes[role] = {
            "initial_mean_rms": initial_mean,
            "final_mean_rms": final_mean,
            "ratio": final_mean / initial_mean,
        }
    final_objective = states[-1]["solve"].final_objective or {}
    components = final_objective.get("components", {})
    return {
        "interpretation": "engineering-only sub-noise smoke test",
        "maximum_homogeneous_response_to_noise": float(homogeneous_snr),
        "maximum_local_response_to_noise": float(local_snr),
        "yield_rmse_reduction_fraction": float((initial_rmse - final_rmse) / initial_rmse),
        "egi_mean_rms_changes": egi_changes,
        "final_objective_components": components,
        "selected_supports": {role: list(window) for role, window in supports},
        "states": state_rows,
        "sensitivity_solves": [{
            "phase_index": int(item["phase_index"]),
            "solve_iteration": int(item["solve_iteration"]),
            "retained_rank": int(item["retained_rank"]),
            "singular_values": np.asarray(item["singular_values"]).tolist(),
            "absolute_singular_threshold": float(item["absolute_singular_threshold"]),
        } for item in sensitivities],
    }


def _summary_page(pdf, summary):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.055, 0.92, "Data-driven EGI identification: local smoke test", fontsize=21, weight="bold")
    supports = summary["selected_supports"]
    components = summary["final_objective_components"]
    lines = [
        "Purpose: verify support selection, objective sensitivity, spatial recovery and BF geometry before workstation trials.",
        "Status: ENGINEERING-ONLY sub-noise smoke; not eligible for scientific interpretation or workstation scale-up.",
        f"Best support-sweep response: homogeneous {summary['maximum_homogeneous_response_to_noise']:.3g} sigma; local {summary['maximum_local_response_to_noise']:.3g} sigma.",
        "",
        "Automatically selected EGI windows",
        *(f"  {role:>6}: {window[0]} × {window[1]} datapoints" for role, window in supports.items()),
        "",
        "Yield-strength recovery",
        *(f"  {row['state']}: RMSE {row['yield_rmse_mpa']:.2f} MPa; range {row['yield_min_mpa']:.1f}–{row['yield_max_mpa']:.1f} MPa" for row in summary["states"]),
        "",
        "Prepared objective",
        *(f"  solve {row['solve_iteration']}: retained rank {row['retained_rank']}; threshold {row['absolute_singular_threshold']:.3g}" for row in summary["sensitivity_solves"]),
        "",
        "Physical-plausibility verdict: DO NOT SCALE UP",
        f"  Positive: BF1 reduces yield-map RMSE by {100.0 * summary['yield_rmse_reduction_fraction']:.1f}% and puts its positive lobe at the neck.",
        "  Concern: the fitted BF is much taller than the specimen and drives the far-field yield strength too low.",
        f"  Concern: terminal component costs are material {components.get('material_cost', float('nan')):.3g}, FRE {components.get('fre_guard_cost', float('nan')):.3g}, broad EGI {components.get('broad_egi_guard_cost', float('nan')):.3g}; roles are not comparably influential.",
        "  Blocking concern: the support responses are far below one noise sigma, so the retained rank and apparent improvement are not scientifically trustworthy.",
        "",
        "Next decision: establish propagated per-observation EGI noise, rerun the support gate, then repeat BF0-BF1 only if at least three supports resolve above the SNR floor.",
    ]
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=112, subsequent_indent="    ") or [""])
    fig.text(0.065, 0.84, "\n".join(wrapped), va="top", family="monospace", fontsize=10.2, linespacing=1.34)
    pdf.savefig(fig)
    plt.close(fig)


def _algorithm_page(pdf, summary):
    """Summarise the exact two-phase algorithm exercised by this run."""
    fig, axis = plt.subplots(figsize=(11.69, 8.27))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    fig.suptitle("Algorithm exercised in this BF0–BF1 smoke test", fontsize=20, weight="bold", y=0.965)
    axis.text(
        0.5, 0.91,
        "ENGINEERING-ONLY: the permissive support gate and provisional scalar noise scales were used only to exercise the workflow.",
        ha="center", va="center", fontsize=10, color="darkred", weight="bold",
    )

    boxes = [
        (
            "1  Phase 0 — homogeneous identification",
            "Fit spatially homogeneous yield strength and hardening modulus.\n"
            "Known elastic properties; SBVF residual; nonlinear least-squares solve.\n"
            "The accepted homogeneous state is the reference for Phase 1.",
        ),
        (
            "2  Build and score an EGI support bank",
            "Generate odd square windows from 3 datapoints to half the minimum specimen bounding-box dimension.\n"
            "At the Phase-0 state, perturb homogeneous yield plus 9 local yield probes by 1% of the allowed range.\n"
            "For every window, record valid coverage, response/noise, singular spectrum and Fisher information.",
        ),
        (
            "3  Select fine, middle and broad EGI supports",
            f"Fine = smallest support passing coverage/local-response gates; broad = largest passing support.\n"
            f"Middle = remaining support adding the most Fisher information:  "
            f"{summary['selected_supports']['fine'][0]}×{summary['selected_supports']['fine'][1]}, "
            f"{summary['selected_supports']['middle'][0]}×{summary['selected_supports']['middle'][1]}, "
            f"{summary['selected_supports']['broad'][0]}×{summary['selected_supports']['broad'][1]} datapoints.\n"
            "This smoke lowered the response gate to 0.005 sigma; the intended scientific gate is at least 1 sigma.",
        ),
        (
            "4  Freeze a sensitivity-informed BF1 objective",
            "Assemble all-frame residuals: FRE plus fine, middle and broad EGI fields; whiten with their noise scales.\n"
            "Finite-difference every active normalised DOF, take the SVD, retain noise-resolved directions, and freeze that\n"
            "projection for the solve. Here 8/8 directions were retained, but that rank is unreliable under sub-noise scaling.",
        ),
        (
            "5  Optimise the first spatial basis function",
            "Start from one EGI-seeded bivariate SPD BF and solve its amplitude/centre/shape together with global parameters.\n"
            "Objective = mean of: robust projected-information cost + FRE stress-scale guard + broad-EGI closure guard.\n"
            "Pattern search used a fixed BF trajectory (30 iterations, 493 evaluations); correction-map BF fitting starts at BF2.",
        ),
    ]
    y_positions = [0.79, 0.63, 0.47, 0.31, 0.15]
    for index, ((title, body), y) in enumerate(zip(boxes, y_positions, strict=True)):
        box = FancyBboxPatch(
            (0.06, y - 0.062), 0.88, 0.118,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor="#eef4f8" if index % 2 == 0 else "#f5f2ea",
            edgecolor="#4d6475", linewidth=1.1,
        )
        axis.add_patch(box)
        axis.text(0.082, y + 0.032, title, fontsize=11.2, weight="bold", va="center")
        axis.text(0.082, y + 0.008, body, fontsize=8.8, va="top", linespacing=1.22)
        if index < len(boxes) - 1:
            axis.annotate(
                "", xy=(0.5, y_positions[index + 1] + 0.067), xytext=(0.5, y - 0.068),
                arrowprops={"arrowstyle": "-|>", "color": "#4d6475", "lw": 1.2},
            )
    pdf.savefig(fig)
    plt.close(fig)


def _extent(experiment):
    x = np.asarray(experiment.specimen_geometry.x)
    y = np.asarray(experiment.specimen_geometry.y)
    return [float(np.nanmin(x)), float(np.nanmax(x)), float(np.nanmin(y)), float(np.nanmax(y))]


def _yield_map_page(pdf, states, truth, mask, experiment):
    extent = _extent(experiment)
    maps = [truth] + [state["maps"]["yield_strength"] for state in states]
    labels = ["Known truth"] + [state["label"] for state in states]
    vmin = float(np.nanpercentile(np.concatenate([item[mask] for item in maps]), 1))
    vmax = float(np.nanpercentile(np.concatenate([item[mask] for item in maps]), 99))
    _map_pages(
        pdf,
        maps,
        labels,
        mask,
        extent,
        title="Yield-strength map by identification state",
        colorbar_label="Yield strength [MPa]",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    # Truth is the reference, so error pages begin with the homogeneous state.
    errors = [100.0 * (item - truth) / truth for item in maps[1:]]
    error_limit = max(1.0, float(np.nanpercentile(np.abs(np.concatenate([item[mask] for item in errors])), 99)))
    _map_pages(
        pdf,
        errors,
        labels[1:],
        mask,
        extent,
        title="Signed yield-strength error by identification state",
        colorbar_label="Error [%]",
        cmap="RdBu_r",
        vmin=-error_limit,
        vmax=error_limit,
    )


def _map_pages(pdf, maps, labels, mask, extent, *, title, colorbar_label, cmap, vmin, vmax):
    """Render specimen maps two per page without changing their physical aspect.

    Matplotlib's ``aspect='equal'`` is the equivalent of MATLAB's ``axis image``:
    an x millimetre occupies the same printed length as a y millimetre.
    """
    for start in range(0, len(maps), 2):
        page_maps = maps[start:start + 2]
        page_labels = labels[start:start + 2]
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 5.3), constrained_layout=True, squeeze=False)
        for axis, values, label in zip(axes[0], page_maps, page_labels, strict=False):
            image = axis.imshow(
                np.where(mask, values, np.nan), origin="lower", extent=extent,
                aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax,
            )
            axis.set_title(label, fontsize=12)
            axis.set(xlabel="x [mm]", ylabel="y [mm]")
        # Keep a stable two-column layout even on an odd final page.
        for axis in axes[0, len(page_maps):]:
            axis.set_visible(False)
        fig.colorbar(image, ax=axes[0, :len(page_maps)], label=colorbar_label, shrink=0.88)
        fig.suptitle(title, fontsize=16)
        pdf.savefig(fig)
        plt.close(fig)


def _egi_page(pdf, states, supports, egi_maps, mask, experiment):
    fig, axes = plt.subplots(len(states), 3, figsize=(11.69, 8.27), constrained_layout=True, squeeze=False)
    extent = _extent(experiment)
    for column, (role, window) in enumerate(supports):
        values = [egi_maps[state["label"]][role] for state in states]
        finite = np.concatenate([item[np.isfinite(item)] for item in values])
        vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
        for row, state in enumerate(states):
            image = axes[row, column].imshow(np.where(mask, values[row], np.nan), origin="lower", extent=extent, aspect="equal", cmap="magma", vmin=0.0, vmax=max(vmax, np.finfo(float).eps))
            axes[row, column].set_title(f"{state['label']}\n{role} EGI {window[0]}×{window[1]}", fontsize=9)
            axes[row, column].set_xlabel("x [mm]")
            axes[row, column].set_ylabel("y [mm]")
        fig.colorbar(image, ax=axes[:, column], shrink=0.7, label="Weighted temporal RMS")
    fig.suptitle("Temporal RMS of the automatically selected EGI maps", fontsize=16)
    pdf.savefig(fig)
    plt.close(fig)


def _sensitivity_pages(pdf, records, experiment):
    extent = _extent(experiment)
    for record in records:
        payload = record
        sensitivity = np.asarray(payload["sensitivity"], dtype=np.float64)
        basis = np.asarray(payload["basis"], dtype=np.float64)
        response = np.sqrt(np.sum(sensitivity**2, axis=1))
        leverage = np.sum(basis**2, axis=1) if basis.size else np.zeros(sensitivity.shape[0])
        maps = _block_maps(payload["blocks"], response, leverage)
        egi = [(name, item) for name, item in maps if item[0].ndim == 2]
        fig, axes = plt.subplots(max(1, len(egi)), 2, figsize=(11.69, 8.27), constrained_layout=True, squeeze=False)
        for row, (name, (response_map, leverage_map)) in enumerate(egi):
            first = axes[row, 0].imshow(response_map, origin="lower", extent=extent, aspect="equal", cmap="viridis")
            second = axes[row, 1].imshow(leverage_map, origin="lower", extent=extent, aspect="equal", cmap="plasma")
            axes[row, 0].set_title(f"{name}: native-DOF response magnitude")
            axes[row, 1].set_title(f"{name}: retained-subspace leverage")
            fig.colorbar(first, ax=axes[row, 0], shrink=0.72)
            fig.colorbar(second, ax=axes[row, 1], shrink=0.72)
            for axis in axes[row]:
                axis.set_xlabel("x [mm]")
                axis.set_ylabel("y [mm]")
        fig.suptitle(f"Sensitivity maps used for solve {payload['solve_iteration']} (rank {payload['retained_rank']})", fontsize=15)
        pdf.savefig(fig)
        plt.close(fig)

        singular = np.asarray(payload["singular_values"], dtype=np.float64)
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 5.3), constrained_layout=True)
        axes[0].semilogy(np.arange(1, singular.size + 1), np.maximum(singular, np.finfo(float).tiny), "o-")
        axes[0].axhline(float(payload["absolute_singular_threshold"]), color="tab:red", linestyle="--", label="retention threshold")
        axes[0].set_xlabel("Singular direction")
        axes[0].set_ylabel("Singular value")
        axes[0].set_title("Noise-resolved sensitivity spectrum")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
        axes[1].bar(np.arange(sensitivity.shape[1]), np.linalg.norm(sensitivity, axis=0))
        axes[1].set_xlabel("Native active DOF")
        axes[1].set_ylabel("Balanced whitened response norm")
        axes[1].set_title("Sensitivity carried by each optimiser coordinate")
        pdf.savefig(fig)
        plt.close(fig)


def _block_maps(blocks, response, leverage):
    output = []
    start = 0
    for block in blocks:
        count = len(block["valid_indices"])
        source_shape = tuple(int(value) for value in block["source_shape"])
        frames = tuple(int(value) for value in block["frame_indices"])
        selected_shape = (len(frames), *source_shape[1:])
        block_response = np.full(int(np.prod(selected_shape)), np.nan)
        block_leverage = np.full(int(np.prod(selected_shape)), np.nan)
        indices = np.asarray(block["valid_indices"], dtype=int)
        block_response[indices] = response[start:start + count]
        block_leverage[indices] = leverage[start:start + count]
        block_response = block_response.reshape(selected_shape)
        block_leverage = block_leverage.reshape(selected_shape)
        response_map = np.sqrt(np.nanmean(block_response**2, axis=0))
        leverage_map = np.sqrt(np.nanmean(block_leverage**2, axis=0))
        output.append((block["name"], (response_map, leverage_map)))
        start += count
    return output


def _basis_page(pdf, states, egi_maps, supports, mask, experiment):
    extent = _extent(experiment)
    broad = "broad"
    # Two states per page keeps both the specimen field and the physical BF
    # geometry readable.  The kernel artists may deliberately enlarge an axis
    # when a fitted BF is larger than the specimen; equal aspect makes that
    # visible without changing its shape.
    for start in range(0, len(states), 2):
        page_states = states[start:start + 2]
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True, squeeze=False)
        for column, state in enumerate(page_states):
            display_limits = _basis_display_limits(state["snapshot"], extent)
            values = state["maps"]["yield_strength"]
            image = axes[0, column].imshow(
                np.where(mask, values, np.nan), origin="lower", extent=extent,
                aspect="equal", cmap="viridis",
            )
            _draw_bases(axes[0, column], state["snapshot"])
            _apply_display_limits(axes[0, column], display_limits)
            axes[0, column].set_title(f"{state['label']} yield map + BFs", fontsize=11)

            target = egi_maps[state["label"]][broad]
            target_image = axes[1, column].imshow(
                np.where(mask, target, np.nan), origin="lower", extent=extent,
                aspect="equal", cmap="magma",
            )
            _draw_bases(axes[1, column], state["snapshot"])
            _apply_display_limits(axes[1, column], display_limits)
            axes[1, column].set_title(f"{state['label']} broad-EGI target + BFs", fontsize=11)
            for axis in axes[:, column]:
                axis.set(xlabel="x [mm]", ylabel="y [mm]")
                axis.set_aspect("equal", adjustable="box")

        for column in range(len(page_states), 2):
            axes[0, column].set_visible(False)
            axes[1, column].set_visible(False)
        fig.colorbar(image, ax=axes[0, :len(page_states)], label="Yield strength [MPa]", shrink=0.78)
        fig.colorbar(target_image, ax=axes[1, :len(page_states)], label="Weighted temporal RMS", shrink=0.78)
        fig.suptitle(
            "Basis geometry by identification state\n"
            "The BF1 solve uses the initial EGI seed; correction-map fitting begins on subsequent refinements",
            fontsize=14,
        )
        pdf.savefig(fig)
        plt.close(fig)


def _basis_display_limits(snapshot, specimen_extent):
    """Return a specimen-focused view, enlarged 20% only on exceeded axes."""
    xmin, xmax, ymin, ymax = specimen_extent
    bf_bounds = [np.inf, -np.inf, np.inf, -np.inf]
    found_basis = False
    for item in snapshot.spatial_parameterisations.get("yield_strength", []):
        if item.summary.get("kind") != "basis_functions":
            continue
        for kernel in item.summary.get("kernels", []):
            found_basis = True
            centre = np.asarray(kernel["centre"], dtype=float)
            radius_x, radius_y = _kernel_axis_radii(kernel)
            bf_bounds[0] = min(bf_bounds[0], centre[0] - radius_x)
            bf_bounds[1] = max(bf_bounds[1], centre[0] + radius_x)
            bf_bounds[2] = min(bf_bounds[2], centre[1] - radius_y)
            bf_bounds[3] = max(bf_bounds[3], centre[1] + radius_y)

    def limit(low, high, basis_low, basis_high):
        if not found_basis or (basis_low >= low and basis_high <= high):
            return low, high
        centre = 0.5 * (low + high)
        half_width = 0.6 * (high - low)  # 120% of specimen bounding-box span.
        return centre - half_width, centre + half_width

    return (
        limit(xmin, xmax, bf_bounds[0], bf_bounds[1]),
        limit(ymin, ymax, bf_bounds[2], bf_bounds[3]),
    )


def _kernel_axis_radii(kernel):
    """Axis-aligned radii of the one-standard-deviation ellipse plotted below."""
    if "covariance" in kernel:
        covariance = np.asarray(kernel["covariance"], dtype=float)
    else:
        variances = np.atleast_1d(np.asarray(kernel["variance"], dtype=float))
        if variances.size == 1:
            variances = np.repeat(variances, 2)
        angle = float(kernel.get("angle", 0.0))
        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        covariance = rotation @ np.diag(variances[:2]) @ rotation.T
    return float(np.sqrt(max(covariance[0, 0], 0.0))), float(np.sqrt(max(covariance[1, 1], 0.0)))


def _apply_display_limits(axis, limits):
    """Clip overlays to the selected specimen-focused field of view."""
    (xmin, xmax), (ymin, ymax) = limits
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)


def _draw_bases(axis, snapshot):
    parameterisations = snapshot.spatial_parameterisations.get("yield_strength", [])
    for item in parameterisations:
        if item.summary.get("kind") != "basis_functions":
            continue
        for index, kernel in enumerate(item.summary.get("kernels", []), start=1):
            centre = np.asarray(kernel["centre"], dtype=float)
            if "covariance" in kernel:
                covariance = np.asarray(kernel["covariance"], dtype=float)
                values, vectors = np.linalg.eigh(covariance)
                order = np.argsort(values)[::-1]
                values = values[order]
                major = vectors[:, order[0]]
                angle = np.degrees(np.arctan2(major[1], major[0]))
            else:
                variances = np.atleast_1d(np.asarray(kernel["variance"], dtype=float))
                if variances.size == 1:
                    variances = np.repeat(variances, 2)
                values = variances
                angle = np.degrees(float(kernel.get("angle", 0.0)))
            height = float(kernel["height"])
            color = "tab:red" if height >= 0 else "tab:blue"
            axis.add_patch(Ellipse(centre, 2*np.sqrt(values[0]), 2*np.sqrt(values[1]), angle=angle, fill=False, color=color, linewidth=2))
            axis.scatter(*centre, marker="x", color=color, s=45)
            axis.text(centre[0], centre[1], f" B{index}", color=color, fontsize=8, weight="bold")


if __name__ == "__main__":
    main()
