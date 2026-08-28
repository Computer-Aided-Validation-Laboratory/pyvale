"""Audit basis geometry and refinement acceptance from an existing VFM run.

This is a read-only Bucket-4 diagnostic. It does not run identification or
modify a saved result. Known synthetic maps are used only for validation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pyvale.vfm import ExperimentData, load_identification_result
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
DEFAULT_RESULT = (
    DATASET
    / "identification/prepared/egi_window_baseline_15500_20260827/"
    "identification_result.yaml"
)
DEFAULT_OUTPUT = Path("dev/vfm/output/bucket_4_basis_acceptance_20260827")
ACTIVITY_FILE = Path(
    "dev/vfm/output/sensitivity_spatial_weighting_checkpoint/"
    "weights_and_activity.npz"
)


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)

    experiment = ExperimentData.load_from_file(_experiment_file(args.input))
    result = load_identification_result(args.result)
    law = load_constitutive_law_from_result(result)
    known = load_known_parameter_maps(args.known_parameters, args.input)
    if known is None:
        raise ValueError("Bucket-4 synthetic validation requires known maps.")

    plasticity = compute_plasticity_diagnostics(experiment, law, known)
    if plasticity is None:
        raise ValueError("Could not calculate known-map plasticity diagnostics.")
    yielded = np.asarray(plasticity.yielded_datapoints, dtype=bool)
    peak_plastic = np.nanmax(plasticity.equivalent_plastic_strain, axis=0)
    high_threshold = float(np.nanpercentile(peak_plastic[yielded], 75.0))
    high_plastic = yielded & (peak_plastic >= high_threshold)

    x = np.asarray(experiment.specimen_geometry.x, dtype=float)
    y = np.asarray(experiment.specimen_geometry.y, dtype=float)
    roi = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(x, y)
    activity = _combined_activity(args.activity)
    low_information = roi & np.isfinite(activity) & (activity <= 0.10)
    high_information = roi & np.isfinite(activity) & (activity >= 0.50)

    phase = result.history.phases[1]
    geometry_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    change_rows: list[dict[str, Any]] = []
    overlap_matrices: dict[int, np.ndarray] = {}
    contribution_maps: dict[int, list[np.ndarray]] = {}

    previous_maps = evaluate_snapshot_parameter_maps(
        result.history.phases[0].final_snapshot,
        experiment,
    )
    previous_cost: float | None = None
    for solve in phase.solve_results:
        if solve.final_snapshot is None:
            continue
        maps = evaluate_snapshot_parameter_maps(solve.final_snapshot, experiment)
        kernels = _extract_yield_kernels(solve.final_snapshot)
        responses = [_kernel_response(kernel, x, y) for kernel in kernels]
        contributions = [
            float(kernel["height_mpa"]) * response
            for kernel, response in zip(kernels, responses, strict=True)
        ]
        contribution_maps[int(solve.solve_iteration)] = contributions
        overlap = _overlap_matrix(responses, roi)
        overlap_matrices[int(solve.solve_iteration)] = overlap

        for index, (kernel, response, contribution) in enumerate(
            zip(kernels, responses, contributions, strict=True)
        ):
            other = np.delete(overlap[index], index) if len(kernels) > 1 else np.array([])
            row = {
                "solve": int(solve.solve_iteration),
                "solve_accepted": bool(solve.accepted),
                "basis": index + 1,
                **kernel,
                "max_response_overlap": (
                    float(np.max(other)) if other.size else 0.0
                ),
                "low_information_energy_fraction": _energy_fraction(
                    contribution, low_information, roi
                ),
                "unyielded_energy_fraction": _energy_fraction(
                    contribution, roi & ~yielded, roi
                ),
                "low_to_high_information_rms_ratio": _rms_ratio(
                    contribution, low_information, high_information
                ),
                "response_area_fraction_above_exp_minus_half": float(
                    np.mean((response >= np.exp(-0.5))[roi])
                ),
            }
            geometry_rows.append(row)

        for first in range(len(kernels)):
            for second in range(first + 1, len(kernels)):
                overlap_rows.append({
                    "solve": int(solve.solve_iteration),
                    "solve_accepted": bool(solve.accepted),
                    "basis_a": first + 1,
                    "basis_b": second + 1,
                    "response_cosine_overlap": float(overlap[first, second]),
                })

        validation = _validate_maps(
            maps,
            known,
            yielded,
            high_plastic,
            roi,
        )
        components = solve.final_objective.get("components", {})
        cost = float(solve.final_objective["cost"])
        validation_rows.append({
            "solve": int(solve.solve_iteration),
            "basis_count": len(kernels),
            "accepted": bool(solve.accepted),
            "status": str(solve.status),
            "evaluations": int(solve.num_evaluations or 0),
            "objective": cost,
            "relative_objective_improvement": (
                None
                if previous_cost is None
                else (previous_cost - cost) / max(abs(previous_cost), 1.0e-12)
            ),
            "equilibrium_gap_cost": _optional_float(
                components.get("equilibrium_gap_cost")
            ),
            "force_cost": _optional_float(components.get("force_cost")),
            **validation,
        })

        delta = maps["yield_strength"] - previous_maps["yield_strength"]
        change_rows.append({
            "solve": int(solve.solve_iteration),
            "basis_count": len(kernels),
            "accepted": bool(solve.accepted),
            "objective": cost,
            "objective_reduction_from_previous_solve": (
                None if previous_cost is None else previous_cost - cost
            ),
            "map_change_rms_roi_mpa": _rms(delta, roi),
            "map_change_rms_low_information_mpa": _rms(delta, low_information),
            "map_change_rms_high_information_mpa": _rms(delta, high_information),
            "map_change_low_information_energy_fraction": _energy_fraction(
                delta, low_information, roi
            ),
            "map_change_unyielded_energy_fraction": _energy_fraction(
                delta, roi & ~yielded, roi
            ),
        })
        previous_maps = maps
        previous_cost = cost

    _write_csv(args.output / "basis_geometry.csv", geometry_rows)
    _write_csv(args.output / "basis_overlap.csv", overlap_rows)
    _write_csv(args.output / "snapshot_validation.csv", validation_rows)
    _write_csv(args.output / "refinement_map_changes.csv", change_rows)
    _plot_geometry(geometry_rows, figures / "basis_geometry.png")
    _plot_validation(validation_rows, figures / "snapshot_validation.png")
    _plot_changes(change_rows, figures / "refinement_map_changes.png")
    _plot_overlaps(overlap_matrices, figures / "basis_overlap_matrices.png")
    _plot_contributions(
        contribution_maps,
        x,
        y,
        figures / "basis_contribution_maps.png",
    )

    summary = {
        "purpose": (
            "Bucket-4 read-only audit of basis geometry, overlap, solve-snapshot "
            "validation, and changes in low-information regions."
        ),
        "result": str(args.result),
        "activity_definition": (
            "Pointwise maximum of the four EGI-29/57 yield-strength and "
            "hardening activity maps after separate 95th-percentile scaling."
        ),
        "low_information_threshold": 0.10,
        "high_information_threshold": 0.50,
        "low_information_roi_fraction": float(np.mean(low_information[roi])),
        "high_information_roi_fraction": float(np.mean(high_information[roi])),
        "known_yielded_roi_fraction": float(np.mean(yielded[roi])),
        "high_plastic_threshold": high_threshold,
        "basis_geometry": geometry_rows,
        "snapshot_validation": validation_rows,
        "refinement_map_changes": change_rows,
        "interpretation_limits": [
            "Known yield/plasticity masks are synthetic validation only.",
            "Sensitivity activity is frozen at the phase-start state.",
            "A solve-to-solve map change includes joint reoptimization of all active DOFs; it is not the isolated effect of only the newest basis.",
            "Kernel-response overlap measures representation redundancy, not objective redundancy.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_report(args.output / "REPORT.md", summary)
    print(json.dumps({
        "output": str(args.output),
        "solves": len(validation_rows),
        "basis_rows": len(geometry_rows),
        "overlap_rows": len(overlap_rows),
    }, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATASET / "prepared")
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--known-parameters", type=Path, default=None)
    parser.add_argument("--activity", type=Path, default=ACTIVITY_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _experiment_file(path: Path) -> Path:
    return path / "experiment_data.yaml" if path.is_dir() else path


def _combined_activity(path: Path) -> np.ndarray:
    with np.load(path) as loaded:
        arrays = [
            np.asarray(loaded[name], dtype=float)
            for name in (
                "egi_29_yield_strength_activity",
                "egi_29_hardening_modulus_activity",
                "egi_57_yield_strength_activity",
                "egi_57_hardening_modulus_activity",
            )
        ]
    scaled = []
    for values in arrays:
        scale = float(np.nanpercentile(values[np.isfinite(values)], 95.0))
        scaled.append(values / max(scale, np.finfo(float).tiny))
    stack = np.stack(scaled)
    finite = np.any(np.isfinite(stack), axis=0)
    combined = np.max(np.where(np.isfinite(stack), stack, -np.inf), axis=0)
    combined[~finite] = np.nan
    return np.clip(combined, 0.0, 1.0)


def _extract_yield_kernels(snapshot: Any) -> list[dict[str, float]]:
    kernels: list[dict[str, float]] = []
    for parameterisation in snapshot.spatial_parameterisations["yield_strength"]:
        if parameterisation.summary.get("kind") != "basis_functions":
            continue
        for kernel in parameterisation.summary.get("kernels", []):
            width_x, width_y = (float(value) for value in kernel["width"])
            angle = float(kernel["angle"])
            if width_y > width_x:
                major, minor = width_y, width_x
                angle += 0.5 * np.pi
            else:
                major, minor = width_x, width_y
            angle = (angle + 0.5 * np.pi) % np.pi - 0.5 * np.pi
            centre_x, centre_y = (float(value) for value in kernel["centre"])
            max_width = float(np.hypot(40.0, 13.0))
            kernels.append({
                "centre_x_mm": centre_x,
                "centre_y_mm": centre_y,
                "height_mpa": float(kernel["height"]),
                "width_x_mm": width_x,
                "width_y_mm": width_y,
                "minor_width_mm": minor,
                "major_width_mm": major,
                "major_angle_deg": float(np.degrees(angle)),
                "centre_x_bound_margin_fraction": min(
                    (centre_x - 50.0) / 40.0, (90.0 - centre_x) / 40.0
                ),
                "centre_y_bound_margin_fraction": min(
                    (centre_y + 6.5) / 13.0, (6.5 - centre_y) / 13.0
                ),
                "minor_width_fraction_of_max": minor / max_width,
                "major_width_fraction_of_max": major / max_width,
                "minor_width_fraction_above_min": (minor - 0.6) / (max_width - 0.6),
                "major_width_fraction_above_min": (major - 0.6) / (max_width - 0.6),
                "absolute_height_fraction_of_bound": abs(float(kernel["height"])) / 1800.0,
            })
    return kernels


def _kernel_response(kernel: dict[str, float], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = x - kernel["centre_x_mm"]
    dy = y - kernel["centre_y_mm"]
    angle = np.radians(kernel["major_angle_deg"])
    local_major = np.cos(angle) * dx + np.sin(angle) * dy
    local_minor = -np.sin(angle) * dx + np.cos(angle) * dy
    return np.exp(-0.5 * (
        (local_major / kernel["major_width_mm"]) ** 2
        + (local_minor / kernel["minor_width_mm"]) ** 2
    ))


def _overlap_matrix(responses: list[np.ndarray], mask: np.ndarray) -> np.ndarray:
    count = len(responses)
    matrix = np.eye(count, dtype=float)
    for first in range(count):
        a = responses[first][mask]
        for second in range(first + 1, count):
            b = responses[second][mask]
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            value = 0.0 if denominator == 0.0 else float(np.dot(a, b) / denominator)
            matrix[first, second] = value
            matrix[second, first] = value
    return matrix


def _validate_maps(
    maps: dict[str, np.ndarray],
    known: dict[str, np.ndarray],
    yielded: np.ndarray,
    high_plastic: np.ndarray,
    roi: np.ndarray,
) -> dict[str, float]:
    identified = np.asarray(maps["yield_strength"], dtype=float)
    truth = np.asarray(known["yield_strength"], dtype=float)

    def metrics(mask: np.ndarray, prefix: str) -> dict[str, float]:
        valid = mask & np.isfinite(identified) & np.isfinite(truth)
        difference = identified[valid] - truth[valid]
        return {
            f"{prefix}_rmse_mpa": float(np.sqrt(np.mean(difference**2))),
            f"{prefix}_mape_percent": float(
                np.mean(np.abs(difference / truth[valid])) * 100.0
            ),
            f"{prefix}_bias_mpa": float(np.mean(difference)),
        }

    hardening = float(np.nanmean(maps["hardening_modulus"]))
    known_hardening = float(np.nanmean(known["hardening_modulus"]))
    return {
        **metrics(roi, "roi"),
        **metrics(yielded, "yielded"),
        **metrics(high_plastic, "high_plastic"),
        **metrics(roi & ~yielded, "unyielded"),
        "hardening_mpa": hardening,
        "hardening_error_mpa": abs(hardening - known_hardening),
    }


def _energy_fraction(values: np.ndarray, subset: np.ndarray, total: np.ndarray) -> float:
    denominator = float(np.sum(np.square(values[total])))
    return 0.0 if denominator == 0.0 else float(
        np.sum(np.square(values[subset])) / denominator
    )


def _rms(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask & np.isfinite(values)]
    return float(np.sqrt(np.mean(np.square(selected))))


def _rms_ratio(values: np.ndarray, numerator: np.ndarray, denominator: np.ndarray) -> float:
    return _rms(values, numerator) / max(_rms(values, denominator), 1.0e-12)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_geometry(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for row in rows:
        marker = "o" if row["solve_accepted"] else "x"
        axes[0].scatter(row["solve"], row["major_width_mm"], marker=marker, color="tab:blue")
        axes[0].scatter(row["solve"], row["minor_width_mm"], marker=marker, color="tab:orange")
        axes[1].scatter(row["centre_x_mm"], row["centre_y_mm"], marker=marker, c=[row["solve"]], cmap="viridis", vmin=0, vmax=max(r["solve"] for r in rows))
        axes[2].scatter(row["solve"], row["max_response_overlap"], marker=marker, color="tab:purple")
    axes[0].axhline(np.hypot(40.0, 13.0), color="black", linestyle="--", label="width upper bound")
    axes[0].axhline(0.6, color="0.5", linestyle=":", label="width lower bound")
    axes[0].set(xlabel="Solve", ylabel="Gaussian width [mm]", yscale="log", title="Major and minor widths")
    axes[0].legend(fontsize=7)
    axes[1].set(xlabel="x [mm]", ylabel="y [mm]", title="Basis centres")
    axes[2].axhline(0.9, color="black", linestyle="--")
    axes[2].set(xlabel="Solve", ylabel="Maximum cosine overlap", ylim=(0, 1.02), title="Within-solve response overlap")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_validation(rows: list[dict[str, Any]], path: Path) -> None:
    solves = [row["solve"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    axes[0].plot(solves, [row["yielded_rmse_mpa"] for row in rows], "o-", label="yielded")
    axes[0].plot(solves, [row["high_plastic_rmse_mpa"] for row in rows], "s-", label="high plastic")
    axes[0].plot(solves, [row["unyielded_rmse_mpa"] for row in rows], "^-", label="unyielded")
    axes[0].set(xlabel="Solve", ylabel="Yield RMSE [MPa]", title="Snapshot map validation")
    axes[0].legend(fontsize=8)
    axes[1].plot(solves, [row["objective"] for row in rows], "o-")
    axes[1].set(xlabel="Solve", ylabel="Objective", title="Optimization objective")
    axes[2].plot(solves, [row["hardening_error_mpa"] for row in rows], "o-", color="tab:red")
    axes[2].set(xlabel="Solve", ylabel="Absolute error [MPa]", title="Hardening validation")
    for axis in axes:
        for row in rows:
            if not row["accepted"]:
                axis.axvline(row["solve"], color="tab:red", alpha=0.15)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_changes(rows: list[dict[str, Any]], path: Path) -> None:
    solves = [row["solve"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot(solves, [row["map_change_rms_low_information_mpa"] for row in rows], "o-", label="low information")
    axes[0].plot(solves, [row["map_change_rms_high_information_mpa"] for row in rows], "s-", label="high information")
    axes[0].set(xlabel="Solve", ylabel="Map-change RMS [MPa]", title="Joint refinement map changes")
    axes[0].legend(fontsize=8)
    axes[1].plot(solves, [100.0 * row["map_change_low_information_energy_fraction"] for row in rows], "o-", label="low information")
    axes[1].plot(solves, [100.0 * row["map_change_unyielded_energy_fraction"] for row in rows], "s-", label="known unyielded")
    axes[1].set(xlabel="Solve", ylabel="Fraction of map-change energy [%]", title="Where refinements change the field")
    axes[1].legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_overlaps(matrices: dict[int, np.ndarray], path: Path) -> None:
    count = len(matrices)
    figure, axes = plt.subplots(1, count, figsize=(3.2 * count, 3.2), squeeze=False, layout="constrained")
    image = None
    for axis, (solve, matrix) in zip(axes[0], matrices.items(), strict=True):
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="magma")
        axis.set(title=f"Solve {solve}", xlabel="Basis", ylabel="Basis")
        axis.set_xticks(range(matrix.shape[0]), range(1, matrix.shape[0] + 1))
        axis.set_yticks(range(matrix.shape[0]), range(1, matrix.shape[0] + 1))
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", color="white", fontsize=7)
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="Cosine overlap")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_contributions(
    maps: dict[int, list[np.ndarray]],
    x: np.ndarray,
    y: np.ndarray,
    path: Path,
) -> None:
    solve = max(maps)
    contributions = maps[solve]
    figure, axes = plt.subplots(1, len(contributions), figsize=(3.4 * len(contributions), 3.2), squeeze=False, layout="constrained")
    maximum = max(float(np.nanmax(np.abs(values))) for values in contributions)
    for index, (axis, values) in enumerate(zip(axes[0], contributions, strict=True)):
        image = axis.pcolormesh(x, y, values, shading="auto", cmap="coolwarm", vmin=-maximum, vmax=maximum)
        axis.set(title=f"Final solve {solve}: basis {index + 1}", xlabel="x [mm]", ylabel="y [mm]")
        figure.colorbar(image, ax=axis, shrink=0.75, label="Contribution [MPa]")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    validation = summary["snapshot_validation"]
    geometry = summary["basis_geometry"]
    final = validation[-1]
    previous = validation[-2]
    widest = max(geometry, key=lambda row: row["major_width_mm"])
    maximum_overlap = max(row["max_response_overlap"] for row in geometry)
    improvement = 100.0 * float(final["relative_objective_improvement"])
    third_rows = [row for row in geometry if row["basis"] == 3]
    latest_third = max(third_rows, key=lambda row: row["solve"])
    fourth_rows = [row for row in geometry if row["basis"] == 4]
    rejected_fourth = (
        max(fourth_rows, key=lambda row: row["solve"])
        if fourth_rows
        else None
    )
    final_state = "accepted" if final["accepted"] else "rejected"
    lines = [
        "# Bucket 4 — basis representation and acceptance audit",
        "",
        "This report is generated from the existing 29/57, smooth-3, 15,500-evaluation baseline. No identification was run.",
        "",
        "## Headline evidence",
        "",
        f"- The widest basis reaches {widest['major_width_mm']:.2f} mm ({100.0 * widest['major_width_fraction_of_max']:.1f}% of its configured upper width).",
        f"- The maximum pairwise kernel-response cosine overlap is {maximum_overlap:.3f}.",
        f"- The {final_state} final solve reduced the objective by {improvement:.2f}% relative to the preceding solve.",
        f"- Yielded RMSE changed from {previous['yielded_rmse_mpa']:.2f} to {final['yielded_rmse_mpa']:.2f} MPa; high-plastic RMSE changed from {previous['high_plastic_rmse_mpa']:.2f} to {final['high_plastic_rmse_mpa']:.2f} MPa.",
        f"- Hardening error changed from {previous['hardening_error_mpa']:.2f} to {final['hardening_error_mpa']:.2f} MPa.",
        f"- The widest basis puts only {100.0 * widest['low_information_energy_fraction']:.1f}% of its contribution energy in the low-information mask and {100.0 * widest['unyielded_energy_fraction']:.1f}% in known-unyielded points.",
        f"- The latest basis 3 puts {100.0 * latest_third['low_information_energy_fraction']:.1f}% of its energy in low-information points and {100.0 * latest_third['unyielded_energy_fraction']:.1f}% in known-unyielded points.",
        "",
        "## Definitions",
        "",
        "Low information is combined phase-start EGI sensitivity activity <= 0.10 after each parameter/window map is scaled by its own 95th percentile. High information is activity >= 0.50. Kernel overlap is cosine similarity of unweighted Gaussian responses over the specimen ROI.",
        "",
        "## Figures",
        "",
        "![Basis geometry and overlap.](figures/basis_geometry.png){width=100%}",
        "",
        "![Accepted and rejected snapshot validation.](figures/snapshot_validation.png){width=100%}",
        "",
        "![Map changes in low-information and unyielded regions.](figures/refinement_map_changes.png){width=100%}",
        "",
        "![Pairwise basis-response overlap.](figures/basis_overlap_matrices.png){width=100%}",
        "",
        "![Individual basis contributions in the final solve.](figures/basis_contribution_maps.png){width=100%}",
        "",
        "## Interpretation limits",
        "",
        *[f"- {item}" for item in summary["interpretation_limits"]],
        "",
    ]
    if rejected_fourth is not None:
        lines[lines.index("## Definitions"):lines.index("## Definitions")] = [
            f"- Rejected basis 4 puts {100.0 * rejected_fourth['low_information_energy_fraction']:.1f}% of its energy in low-information points, {100.0 * rejected_fourth['unyielded_energy_fraction']:.1f}% in known-unyielded points, and overlaps another basis by {rejected_fourth['max_response_overlap']:.3f}.",
            "",
            "## Preliminary checkpoint decision",
            "",
            "The near-upper-bound broad basis is not currently the strongest source of low-information tail error; its contribution remains concentrated in the active/yielded zone. Width control is therefore not justified yet.",
            "",
            "The stronger concern is objective-to-map alignment and overlapping low-information structure. The accepted third basis substantially changes known-unyielded points, while the rejected fourth basis overlaps it and has the opposite sign. Lowering the 5% gate would accept this particular fourth solve even though yielded RMSE, yielded MAPE, ROI error, and hardening error worsen; only high-plastic RMSE improves slightly. This is not evidence of premature rejection.",
            "",
            "Next gate: measure optimizer/path repeatability at the retained three-basis model order. Do not change the acceptance threshold or width bounds until the size of that repeatability band is known.",
            "",
        ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
