"""Read-only objective-discrimination and optimiser-coordinate audit.

This script evaluates saved synthetic states only.  It neither runs an
identification nor changes its configuration.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyvale.vfm import (
    EquilibriumGapMetric,
    ExperimentData,
    SliceConfig,
    SliceWiseForceReconstructionMetric,
    load_identification_result,
)
from pyvale.vfm.postprocessing import (
    compute_plasticity_diagnostics,
    load_constitutive_law_from_result,
    load_known_parameter_maps,
)


DATASET = Path(
    "/home/robh/1_Projects/pyvale-vfm-test-data/notched-ebw/"
    "synthetic-fe/wdbn1-idealised-yield/pyvale-vfm"
)
CONTROL = DATASET / (
    "identification/prepared/egi_window_baseline_15500_20260827/"
    "identification_result.yaml"
)
OUTPUT = Path("dev/vfm/output/notched_ebw_objective_discrimination_20260828")
WINDOWS = (29, 57)
WINDOW_WEIGHTS = np.asarray(WINDOWS, dtype=float) / sum(WINDOWS)
FORCE_WEIGHT = 0.1
SENSITIVITY_STEP = 1.0e-3


@dataclass(frozen=True)
class Direction:
    label: str
    kind: str
    lower: float
    upper: float
    scaling: str
    target: str | None = None
    kernel: dict[str, float] | None = None
    attribute: str | None = None

    def value(self) -> float:
        if self.kind == "homogeneous":
            assert self.kernel is not None
            return float(self.kernel["value"])
        if self.kind == "height":
            assert self.kernel is not None
            return float(self.kernel["height"])
        assert self.kernel is not None and self.attribute is not None
        return float(self.kernel[self.attribute])


def main() -> None:
    args = _args()
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)

    experiment = ExperimentData.load_from_file(args.input / "experiment_data.yaml")
    control = load_identification_result(args.control)
    known = load_known_parameter_maps(args.input / "known_parameter_maps.npz")
    if known is None:
        raise RuntimeError("Known maps are required for this synthetic diagnostic.")
    law = load_constitutive_law_from_result(control)
    metrics = _metrics(experiment)
    baselines = _baselines(control)
    truth = {name: np.asarray(value, dtype=float) for name, value in known.items()}
    identified = {name: np.asarray(value, dtype=float) for name, value in control.parameter_maps.items()}

    truth_eval = _evaluate(law, truth, experiment, metrics, baselines)
    identified_eval = _evaluate(law, identified, experiment, metrics, baselines)
    sensitivity = _yield_sensitivity(law, truth, experiment, metrics, baselines)
    plasticity = compute_plasticity_diagnostics(experiment, law, truth)
    if plasticity is None:
        raise RuntimeError("Constitutive law does not expose plasticity diagnostics.")

    objective = _objective_discrimination(
        truth_eval, identified_eval, sensitivity,
        np.asarray(plasticity.equivalent_plastic_strain, dtype=float),
        metrics[1], experiment,
    )
    _write_objective_outputs(args.output, figures, objective, experiment, baselines)

    directions = _directions(control, experiment)
    exact = _exact_coordinate_jacobian(
        law, identified, experiment, metrics, baselines, directions,
    )
    _write_jacobian_outputs(args.output, figures, exact, directions)
    _write_report(args.output, objective, exact, directions, baselines)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATASET / "prepared")
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def _metrics(experiment):
    egi = [EquilibriumGapMetric(window_size=(size, size)) for size in WINDOWS]
    for metric in egi:
        metric.initialise(experiment)
    force = SliceWiseForceReconstructionMetric(
        slice_config=SliceConfig(axis="x", num_slices=63)
    )
    force.initialise(experiment)
    return egi, force


def _baselines(control) -> tuple[np.ndarray, float]:
    accepted = [item for item in control.history.phases[1].solve_results if item.accepted]
    components = accepted[-1].final_objective["components"]
    return np.asarray(components["egi_baselines"], dtype=float), float(components["force_baseline"])


def _evaluate(law, maps, experiment, metrics, baselines):
    stress = law.calculate_stress(experiment.strain, maps)
    egi_results = [metric.evaluate_equilibrium_gap(stress) for metric in metrics[0]]
    force_result = metrics[1].evaluate_force_recon_error(stress, experiment)
    blocks = []
    for index, result in enumerate(egi_results):
        values = np.asarray(result.normalised_gap, dtype=float)
        temporal = np.asarray(result.metric_result.additional_fields["temporal_weights"], dtype=float)
        valid = np.isfinite(values)
        residual = values * np.sqrt(temporal)[:, None, None]
        residual /= np.sqrt(np.count_nonzero(valid)) * baselines[0][index]
        residual[~valid] = np.nan
        blocks.append(residual)
    metadata = force_result.metric_result.additional_fields
    force_values = np.asarray(metadata["normalised_residual"], dtype=float)
    force_residual = force_values * np.sqrt(np.asarray(metadata["temporal_weights"], dtype=float))[:, None]
    force_residual *= np.sqrt(np.asarray(metadata["spatial_weights"], dtype=float))[None, :]
    force_residual /= baselines[1]
    blocks.append(force_residual)
    coefficients = np.asarray([
        (1.0 - FORCE_WEIGHT) * WINDOW_WEIGHTS[0],
        (1.0 - FORCE_WEIGHT) * WINDOW_WEIGHTS[1],
        FORCE_WEIGHT,
    ])
    norms = np.asarray([np.sqrt(np.nansum(block**2)) for block in blocks])
    return {"blocks": blocks, "coefficients": coefficients, "norms": norms,
            "objective": float(np.dot(coefficients, norms))}


def _yield_sensitivity(law, truth, experiment, metrics, baselines):
    # This is the truth-state response to one unit of the normalised homogeneous
    # yield coordinate, i.e. the coordinate used by the optimiser for this DOF.
    span = 2000.0 - 200.0
    plus = {name: value.copy() for name, value in truth.items()}
    minus = {name: value.copy() for name, value in truth.items()}
    plus["yield_strength"] += SENSITIVITY_STEP * span
    minus["yield_strength"] -= SENSITIVITY_STEP * span
    upper = _evaluate(law, plus, experiment, metrics, baselines)
    lower = _evaluate(law, minus, experiment, metrics, baselines)
    return [np.abs((high - low) / (2.0 * SENSITIVITY_STEP)) for high, low in zip(upper["blocks"], lower["blocks"], strict=True)]


def _objective_discrimination(truth, identified, sensitivity, plastic, force_metric, experiment):
    attribution, energy_delta, truth_attribution, identified_attribution = [], [], [], []
    for truth_block, identified_block, coefficient in zip(
        truth["blocks"], identified["blocks"], truth["coefficients"], strict=True
    ):
        truth_norm = np.sqrt(np.nansum(truth_block**2))
        identified_norm = np.sqrt(np.nansum(identified_block**2))
        # This exact attribution sums to the scalar-objective difference even
        # though each metric enters it through an RMS norm.
        attribution.append(coefficient * (
            identified_block**2 / identified_norm - truth_block**2 / truth_norm
        ))
        energy_delta.append(identified_block**2 - truth_block**2)
        truth_attribution.append(coefficient * truth_block**2 / truth_norm)
        identified_attribution.append(coefficient * identified_block**2 / identified_norm)

    partition = force_metric.slice_partition
    assert partition is not None
    force_plastic = np.full_like(attribution[2], np.nan)
    force_peak = np.full_like(attribution[2], np.nan)
    for index in range(partition.num_slices):
        mask = partition.get_slice_mask(index)
        force_plastic[:, index] = np.nanmean(plastic[:, mask], axis=1)
        force_peak[:, index] = np.nanmax(plastic[:, mask], axis=1)
    return {
        "attribution": attribution,
        "energy_delta": energy_delta,
        "truth_attribution": truth_attribution,
        "identified_attribution": identified_attribution,
        "sensitivity": sensitivity,
        "plastic": [plastic, plastic, force_plastic],
        "plastic_peak": [plastic, plastic, force_peak],
        "objective_truth": truth["objective"],
        "objective_identified": identified["objective"],
        "slice_centres": partition.centres,
        "slice_boundaries": partition.boundaries,
        "concentration": _concentration(attribution, truth_attribution, sensitivity, [plastic, plastic, force_plastic]),
    }


def _concentration(attribution, truth_attribution, sensitivity, plastic) -> dict[str, float]:
    delta = np.concatenate([item[np.isfinite(item)] for item in attribution])
    response = np.concatenate([
        item[np.isfinite(attribution[index])]
        for index, item in enumerate(sensitivity)
    ])
    strain = np.concatenate([
        item[np.isfinite(attribution[index])]
        for index, item in enumerate(plastic)
    ])
    positive = np.maximum(delta, 0.0)
    truth_local = np.concatenate([
        item[np.isfinite(attribution[index])]
        for index, item in enumerate(truth_attribution)
    ])
    total_positive = float(np.sum(positive))
    summary = {
        "yielded_observation_fraction": float(np.mean(strain > 1.0e-14)),
        "yielded_positive_gap_fraction": float(np.sum(positive[strain > 1.0e-14]) / total_positive),
        "unyielded_truth_objective_fraction": float(np.sum(truth_local[strain <= 1.0e-14]) / np.sum(truth_local)),
    }
    for fraction in (0.05, 0.10, 0.20):
        count = max(1, round(fraction * delta.size))
        selected = np.argsort(response)[-count:]
        summary[f"top_{int(fraction * 100)}pct_sensitivity_positive_gap_fraction"] = float(np.sum(positive[selected]) / total_positive)
    return summary


def _write_objective_outputs(output, figures, values, experiment, baselines) -> None:
    payload = {
        "egi_29_delta_j": values["attribution"][0],
        "egi_57_delta_j": values["attribution"][1],
        "fre_delta_j": values["attribution"][2],
        "egi_29_delta_r_squared": values["energy_delta"][0],
        "egi_57_delta_r_squared": values["energy_delta"][1],
        "fre_delta_r_squared": values["energy_delta"][2],
        "egi_29_truth_j": values["truth_attribution"][0],
        "egi_57_truth_j": values["truth_attribution"][1],
        "fre_truth_j": values["truth_attribution"][2],
        "egi_29_yield_sensitivity": values["sensitivity"][0],
        "egi_57_yield_sensitivity": values["sensitivity"][1],
        "fre_yield_sensitivity": values["sensitivity"][2],
        "equivalent_plastic_strain": values["plastic"][0],
        "fre_mean_equivalent_plastic_strain": values["plastic"][2],
        "fre_peak_equivalent_plastic_strain": values["plastic_peak"][2],
        "x": experiment.specimen_geometry.x,
        "y": experiment.specimen_geometry.y,
        "time": experiment.timesteps,
        "slice_centres": values["slice_centres"],
        "slice_boundaries": values["slice_boundaries"],
        "egi_baselines": baselines[0],
        "force_baseline": np.asarray(baselines[1]),
    }
    np.savez_compressed(output / "objective_discrimination.npz", **payload)
    _plot_egi_small_multiples(values["attribution"][0], experiment, figures / "egi_29_delta_j.png", "EGI-29 exact objective attribution difference")
    _plot_egi_small_multiples(values["attribution"][1], experiment, figures / "egi_57_delta_j.png", "EGI-57 exact objective attribution difference")
    _plot_fre(values["attribution"][2], experiment.timesteps, values["slice_centres"], figures / "fre_delta_j.png")
    _plot_cumulative(values, figures / "cumulative_delta_j.png")


def _plot_egi_small_multiples(data, experiment, path, title) -> None:
    limit = np.nanpercentile(np.abs(data), 99.0)
    figure, axes = plt.subplots(2, 7, figsize=(15, 4.7), layout="constrained")
    for index, axis in enumerate(axes.flat):
        image = axis.pcolormesh(experiment.specimen_geometry.x, experiment.specimen_geometry.y, data[index], shading="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(f"t={experiment.timesteps[index]:.3f}", fontsize=8)
        axis.set_aspect("equal")
        axis.tick_params(labelsize=6)
    figure.colorbar(image, ax=axes, label=r"$\Delta j_i$")
    figure.suptitle(title)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_fre(data, time, centres, path) -> None:
    figure, axis = plt.subplots(figsize=(8, 3.6), layout="constrained")
    limit = np.nanpercentile(np.abs(data), 99.0)
    image = axis.pcolormesh(centres, time, data, shading="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    axis.set(xlabel="slice centre x [mm]", ylabel="time", title="FRE exact objective attribution difference")
    figure.colorbar(image, ax=axis, label=r"$\Delta j_i$")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_cumulative(values, path) -> None:
    delta = np.concatenate([np.ravel(item[np.isfinite(item)]) for item in values["attribution"]])
    plastic = np.concatenate([np.ravel(item[np.isfinite(values["attribution"][index])]) for index, item in enumerate(values["plastic"])])
    sensitivity = np.concatenate([np.ravel(item[np.isfinite(values["attribution"][index])]) for index, item in enumerate(values["sensitivity"])])
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for axis, coordinate, label in zip(axes, (plastic, sensitivity), ("accumulated equivalent plastic strain", "yield sensitivity"), strict=True):
        order = np.argsort(coordinate)
        cumulative = np.cumsum(delta[order])
        axis.plot(np.linspace(0.0, 1.0, cumulative.size), cumulative, label="signed cumulative")
        axis.plot(np.linspace(0.0, 1.0, cumulative.size), np.cumsum(np.maximum(delta[order], 0.0)), label="positive-only")
        axis.set(xlabel=f"fraction ordered by {label}", ylabel=r"cumulative $\Delta J$")
        axis.legend(fontsize=7)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _directions(control, experiment) -> list[Direction]:
    x, y = experiment.specimen_geometry.x, experiment.specimen_geometry.y
    spacing = min(float(np.nanmedian(np.diff(x, axis=1))), float(np.nanmedian(np.diff(y, axis=0))))
    variance_lower = (3.0 * spacing) ** 2
    variance_upper = float(np.hypot(np.ptp(x), np.ptp(y)) ** 2)
    kernels = []
    homogeneous = {}
    for snapshot in control.history.phases[1].final_snapshot.spatial_parameterisations["yield_strength"]:
        if snapshot.summary.get("kind") == "homogeneous":
            homogeneous["yield_strength"] = float(snapshot.summary["value"])
        elif snapshot.summary.get("kind") == "basis_functions":
            for item in snapshot.summary["kernels"]:
                kernels.append({"centre_x": float(item["centre"][0]), "centre_y": float(item["centre"][1]), "variance_x": float(item["variance"][0]), "variance_y": float(item["variance"][1]), "angle": float(item["angle"]), "height": float(item["height"])})
    for snapshot in control.history.phases[1].final_snapshot.spatial_parameterisations["hardening_modulus"]:
        if snapshot.summary.get("kind") == "homogeneous":
            homogeneous["hardening_modulus"] = float(snapshot.summary["value"])
    result = [
        Direction("yield_homogeneous", "homogeneous", 200.0, 2000.0, "linear", "yield_strength", {"value": homogeneous["yield_strength"]}),
        Direction("hardening_homogeneous", "homogeneous", 500.0, 10000.0, "linear", "hardening_modulus", {"value": homogeneous["hardening_modulus"]}),
    ]
    for index, kernel in enumerate(kernels, start=1):
        result.append(Direction(f"basis_{index}_height", "height", -1800.0, 1800.0, "linear", kernel=kernel))
        result.extend([
            Direction(f"basis_{index}_centre_x", "geometry", float(np.nanmin(x)), float(np.nanmax(x)), "linear", kernel=kernel, attribute="centre_x"),
            Direction(f"basis_{index}_centre_y", "geometry", float(np.nanmin(y)), float(np.nanmax(y)), "linear", kernel=kernel, attribute="centre_y"),
            Direction(f"basis_{index}_variance_x", "geometry", variance_lower, variance_upper, "log", kernel=kernel, attribute="variance_x"),
            Direction(f"basis_{index}_variance_y", "geometry", variance_lower, variance_upper, "log", kernel=kernel, attribute="variance_y"),
            Direction(f"basis_{index}_angle", "geometry", -0.5 * np.pi, 0.5 * np.pi, "linear", kernel=kernel, attribute="angle"),
        ])
    return result


def _normalise(value, lower, upper, scaling) -> float:
    if scaling == "log":
        return float((np.log(value) - np.log(lower)) / (np.log(upper) - np.log(lower)))
    return float((value - lower) / (upper - lower))


def _denormalise(value, lower, upper, scaling) -> float:
    if scaling == "log":
        return float(np.exp(np.log(lower) + value * (np.log(upper) - np.log(lower))))
    return float(lower + value * (upper - lower))


def _kernel(kernel, x, y):
    dx, dy = x - kernel["centre_x"], y - kernel["centre_y"]
    cosine, sine = np.cos(kernel["angle"]), np.sin(kernel["angle"])
    return np.exp(-0.5 * (((cosine * dx + sine * dy) ** 2 / kernel["variance_x"]) + ((-sine * dx + cosine * dy) ** 2 / kernel["variance_y"])))


def _perturb_maps(maps, direction, coordinate, experiment):
    result = {name: value.copy() for name, value in maps.items()}
    physical = _denormalise(coordinate, direction.lower, direction.upper, direction.scaling)
    current = direction.value()
    if direction.kind == "homogeneous":
        assert direction.target is not None
        result[direction.target] += physical - current
        return result
    assert direction.kernel is not None
    x, y = experiment.specimen_geometry.x, experiment.specimen_geometry.y
    if direction.kind == "height":
        result["yield_strength"] += (physical - current) * _kernel(direction.kernel, x, y)
        return result
    changed = dict(direction.kernel)
    assert direction.attribute is not None
    changed[direction.attribute] = physical
    result["yield_strength"] += direction.kernel["height"] * (_kernel(changed, x, y) - _kernel(direction.kernel, x, y))
    return result


def _flatten_blocks(blocks) -> np.ndarray:
    """Match the valid-only ordering used by the saved investigation Jacobian."""
    return np.concatenate([block[np.isfinite(block)] for block in blocks])


def _exact_coordinate_jacobian(law, maps, experiment, metrics, baselines, directions):
    base = _evaluate(law, maps, experiment, metrics, baselines)
    columns, stability = [], []
    for direction in directions:
        z = _normalise(direction.value(), direction.lower, direction.upper, direction.scaling)
        # One bounded forward/backup difference per direction keeps this
        # read-only audit practical; the saved multi-step linear audit remains
        # available for stability context.
        other = z + SENSITIVITY_STEP if z + SENSITIVITY_STEP <= 1.0 else z - SENSITIVITY_STEP
        side = _evaluate(law, _perturb_maps(maps, direction, other, experiment), experiment, metrics, baselines)
        estimate = _flatten_blocks([(a - b) / (other - z) for a, b in zip(side["blocks"], base["blocks"], strict=True)])
        columns.append(estimate)
        stability.append({"label": direction.label, "coordinate": z, "step": SENSITIVITY_STEP, "stable": None})
    jacobian = np.column_stack(columns)
    return {"jacobian": jacobian, "labels": [item.label for item in directions], "stability": stability}


def _svd(values):
    _, singular, right = np.linalg.svd(values, full_matrices=False)
    norms = np.linalg.norm(values, axis=0)
    correlation = values.T @ values / np.maximum(np.outer(norms, norms), 1e-300)
    return {"singular_values": singular.tolist(), "condition_number": float(singular[0] / max(singular[-1], 1e-300)), "ranks": {str(threshold): int(np.count_nonzero(singular / singular[0] >= threshold)) for threshold in (1e-2, 1e-3, 1e-4)}, "right_vectors": right.tolist(), "column_correlation": correlation.tolist()}


def _write_jacobian_outputs(output, figures, result, directions) -> None:
    original_path = Path("dev/vfm/output/notched_ebw_identifiability_20260827/jacobian_full/identified_seed0.npz")
    with np.load(original_path) as saved:
        original = np.asarray(saved["jacobian"], dtype=float)
        original_labels = [str(item) for item in saved["labels"]]
    if original_labels != result["labels"]:
        raise RuntimeError("Saved and exact-coordinate Jacobian labels differ.")
    analytic = original.copy()
    for index, direction in enumerate(directions):
        if direction.scaling == "log":
            factor = direction.value() * np.log(direction.upper / direction.lower) / (direction.upper - direction.lower)
            analytic[:, index] *= factor
    direct_svd, analytic_svd, old_svd = _svd(result["jacobian"]), _svd(analytic), _svd(original)
    np.savez_compressed(output / "exact_coordinate_jacobian.npz", jacobian=result["jacobian"], analytic_chain_rule_jacobian=analytic, old_linear_jacobian=original, labels=np.asarray(result["labels"]))
    (output / "exact_coordinate_jacobian_summary.json").write_text(json.dumps({"direct": direct_svd, "analytic_chain_rule": analytic_svd, "old_linear": old_svd, "stability": result["stability"]}, indent=2))
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for values, label in ((old_svd, "old linear variance"), (analytic_svd, "chain-rule log"), (direct_svd, "direct optimiser coordinate")):
        singular = np.asarray(values["singular_values"])
        axes[0].semilogy(np.arange(1, singular.size + 1), singular / singular[0], "o-", ms=3, label=label)
    axes[0].set(xlabel="mode", ylabel=r"$\sigma_i/\sigma_1$", title="SVD comparison")
    axes[0].legend(fontsize=7)
    image = axes[1].imshow(np.asarray(direct_svd["column_correlation"]), vmin=-1, vmax=1, cmap="coolwarm")
    axes[1].set(title="direct-coordinate column correlation")
    figure.colorbar(image, ax=axes[1])
    figure.savefig(figures / "exact_coordinate_svd.png", dpi=180)
    plt.close(figure)


def _write_report(output, objective, exact, directions, baselines) -> None:
    delta = objective["objective_identified"] - objective["objective_truth"]
    by_block = [float(np.nansum(item)) for item in objective["attribution"]]
    direct = _svd(exact["jacobian"])
    lines = [
        "# Notched-EBW objective discrimination and exact-coordinate audit",
        "",
        "## Closure",
        "",
        f"- Truth objective: {objective['objective_truth']:.8f}",
        f"- Identified objective: {objective['objective_identified']:.8f}",
        f"- Objective gap: {delta:.8f}",
        f"- Sum of spatial-temporal attributions: {sum(by_block):.8f}",
        f"- EGI-29 / EGI-57 / FRE attribution: {by_block[0]:.8f}, {by_block[1]:.8f}, {by_block[2]:.8f}",
        "",
        "## Discrimination concentration",
        "",
        f"- Yielded observations: {100.0 * objective['concentration']['yielded_observation_fraction']:.2f}% of metric rows.",
        f"- Truth objective attributed to unyielded observations: {100.0 * objective['concentration']['unyielded_truth_objective_fraction']:.2f}%.",
        f"- Positive objective gap in yielded observations: {100.0 * objective['concentration']['yielded_positive_gap_fraction']:.2f}%.",
        f"- Positive gap captured by top 5% / 10% / 20% yield-sensitive rows: {100.0 * objective['concentration']['top_5pct_sensitivity_positive_gap_fraction']:.2f}% / {100.0 * objective['concentration']['top_10pct_sensitivity_positive_gap_fraction']:.2f}% / {100.0 * objective['concentration']['top_20pct_sensitivity_positive_gap_fraction']:.2f}%.",
        "",
        "## Exact optimiser-coordinate Jacobian",
        "",
        "- Coordinates use the production bounds and log scaling for all Gaussian variances.",
        f"- Condition number: {direct['condition_number']:.6g}",
        f"- Ranks: {direct['ranks']}",
        f"- Direct finite-difference step: {SENSITIVITY_STEP:g} in normalised optimiser coordinates.",
        "",
        "## Coordinate table",
        "",
        "| Parameter | Value | Lower | Upper | Scaling | Normalised value |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for direction in directions:
        lines.append(f"| {direction.label} | {direction.value():.8g} | {direction.lower:.8g} | {direction.upper:.8g} | {direction.scaling} | {_normalise(direction.value(), direction.lower, direction.upper, direction.scaling):.8g} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
