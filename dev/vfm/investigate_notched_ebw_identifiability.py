"""Synthetic notched-EBW closure and dimensionless residual-Jacobian audit.

Development-only diagnostic. It never runs identification or alters VFM
production settings. Native FE centroid stress is mapped to the exact
prepared VFM grid solely to quantify the closure floor.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from pyvale.vfm import EquilibriumGapMetric, ExperimentData, SliceConfig, SliceWiseForceReconstructionMetric, load_identification_result
from pyvale.vfm.postprocessing import load_constitutive_law_from_result, load_known_parameter_maps

DATASET = Path("/home/robh/1_Projects/pyvale-vfm-test-data/notched-ebw/synthetic-fe/wdbn1-idealised-yield/pyvale-vfm")
CASE = DATASET.parent
CONTROL = DATASET / "identification/prepared/egi_window_baseline_15500_20260827/identification_result.yaml"
ORACLE = Path("dev/vfm/output/notched_ebw_capacity_checks/roi_min1_max5/fitted_maps.npz")
ORACLE_PARAMETERS = Path("dev/vfm/output/notched_ebw_capacity_checks/roi_min1_max5/basis_parameters.csv")
OUTPUT = Path("dev/vfm/output/notched_ebw_identifiability_20260827")
WINDOWS = (29, 57)
WINDOW_WEIGHTS = np.asarray(WINDOWS, dtype=float) / sum(WINDOWS)
FORCE_WEIGHT = 0.1
STEPS = (5e-4, 1e-3, 2e-3)


@dataclass(frozen=True)
class State:
    name: str
    maps: dict[str, np.ndarray]


@dataclass(frozen=True)
class Direction:
    label: str
    kind: str
    target: str
    span: float
    kernel: dict[str, float] | None = None
    attribute: str | None = None


def main() -> None:
    args = _args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "cache_v2"
    cache.mkdir(exist_ok=True)
    (args.output / "figures").mkdir(exist_ok=True)
    experiment = ExperimentData.load_from_file(args.input / "experiment_data.yaml")
    control = load_identification_result(args.control)
    known = load_known_parameter_maps(args.input / "known_parameter_maps.npz")
    if known is None:
        raise RuntimeError("Known maps are required for this synthetic diagnostic.")
    metrics = _metrics(experiment)
    baselines = _baselines(control)
    states = _states(known, args.oracle, control)
    closure = {state.name: _evaluate_maps(state.name, state.maps, experiment, control, metrics, baselines, cache) for state in states}
    _write_closure(args.output, closure, control)
    if args.stage == "closure":
        return
    common_output = args.output / "jacobian_common"
    if args.stage in {"all", "common"}:
        common = _jacobian_suite([s for s in states if s.name in {"truth", "oracle_5basis", "identified_seed0"}], _common_directions(experiment), experiment, control, metrics, baselines, cache, common_output)
    else:
        common = json.loads((common_output / "summary.json").read_text())
    if args.stage == "common":
        _write_report(args.output, closure, {"common": common}, None)
        return
    full = _jacobian_suite([next(s for s in states if s.name == "identified_seed0")], _full_directions(control, experiment), experiment, control, metrics, baselines, cache, args.output / "jacobian_full")
    fe = _fe_closure(experiment, metrics, baselines, cache, args.output / "fe_closure") if args.stage in {"all", "fe-closure"} else None
    _write_report(args.output, closure, {"common": common, "full": full}, fe)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATASET / "prepared")
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--oracle", type=Path, default=ORACLE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--stage", choices=("all", "closure", "common", "full", "fe-closure"), default="all")
    return parser.parse_args()


def _metrics(experiment):
    egi = [EquilibriumGapMetric(window_size=(size, size)) for size in WINDOWS]
    for item in egi:
        item.initialise(experiment)
    force = SliceWiseForceReconstructionMetric(slice_config=SliceConfig(axis="x", num_slices=63))
    force.initialise(experiment)
    return egi, force


def _baselines(control):
    accepted = [solve for solve in control.history.phases[1].solve_results if solve.accepted]
    component = accepted[-1].final_objective["components"]
    return np.asarray(component["egi_baselines"], dtype=float), float(component["force_baseline"])


def _states(known, oracle_path, control):
    truth = {name: np.asarray(value, dtype=float).copy() for name, value in known.items()}
    with np.load(oracle_path) as data:
        counts = np.asarray(data["requested_basis_counts"], dtype=int)
        oracle_yield = np.asarray(data["fitted_yield_strength"][np.where(counts == 5)[0][0]], dtype=float)
    identified = {name: np.asarray(value, dtype=float).copy() for name, value in control.parameter_maps.items()}
    return [State("truth", truth), State("oracle_5basis", truth | {"yield_strength": oracle_yield}), State("identified_seed0", identified)]


def _evaluate_maps(name, maps, experiment, control, metrics, baselines, cache):
    path = cache / f"residual_{name}.npz"
    if path.exists():
        try:
            with np.load(path) as data:
                return {key: data[key] for key in data.files}
        except (OSError, ValueError, EOFError):
            path.unlink()
    stress = load_constitutive_law_from_result(control).calculate_stress(experiment.strain, maps)
    return _evaluate_stress(name, stress, maps, experiment, metrics, baselines, cache)


def _evaluate_stress(name, stress, maps, experiment, metrics, baselines, cache):
    path = cache / f"residual_{name}.npz"
    if path.exists():
        try:
            with np.load(path) as data:
                return {key: data[key] for key in data.files}
        except (OSError, ValueError, EOFError):
            path.unlink()
    egi = [metric.evaluate_equilibrium_gap(stress) for metric in metrics[0]]
    force = metrics[1].evaluate_force_recon_error(stress, experiment)
    residual, blocks = _residual_vector(egi, force, baselines, production=False)
    production, _ = _residual_vector(egi, force, baselines, production=True)
    egi_scalars = np.asarray([item.weighted_spatiotemporal_rms for item in egi], dtype=float)
    force_scalar = float(force.weighted_spatiotemporal_rms)
    objective = (1 - FORCE_WEIGHT) * float(np.dot(WINDOW_WEIGHTS, egi_scalars / baselines[0])) + FORCE_WEIGHT * force_scalar / baselines[1]
    # Residuals are enough for resume/SVD.  Caching every 14-step stress field
    # for every finite-difference point is needlessly large.
    payload = {"residual": residual, "residual_production": production, "egi_scalars": egi_scalars, "force_scalar": np.asarray(force_scalar), "objective": np.asarray(objective), **{f"block_{key}": value for key, value in blocks.items()}}
    if maps:
        payload["yield_mean"] = np.asarray(np.nanmean(maps["yield_strength"]))
        payload["hardening_mean"] = np.asarray(np.nanmean(maps["hardening_modulus"]))
    np.savez_compressed(path, **payload)
    return payload


def _residual_vector(egi, force, baselines, *, production):
    blocks = {}
    for index, item in enumerate(egi):
        values = np.asarray(item.normalised_gap, dtype=float)
        temporal = np.asarray(item.metric_result.additional_fields["temporal_weights"], dtype=float)
        valid = np.isfinite(values)
        block = (values * np.sqrt(temporal)[:, None, None] / np.sqrt(np.count_nonzero(valid)))[valid] / baselines[0][index]
        if production:
            block *= np.sqrt((1 - FORCE_WEIGHT) * WINDOW_WEIGHTS[index])
        blocks[f"egi_{WINDOWS[index]}"] = block
    values = np.asarray(force.metric_result.additional_fields["normalised_residual"], dtype=float)
    temporal = np.asarray(force.metric_result.additional_fields["temporal_weights"], dtype=float)
    spatial = np.asarray(force.metric_result.additional_fields["spatial_weights"], dtype=float)
    block = (values * np.sqrt(temporal)[:, None] * np.sqrt(spatial)[None, :]).ravel() / baselines[1]
    if production:
        block *= np.sqrt(FORCE_WEIGHT)
    blocks["fre"] = block
    return np.concatenate(list(blocks.values())), blocks


def _bounds(experiment):
    x, y = experiment.specimen_geometry.x, experiment.specimen_geometry.y
    spacing = min(float(np.nanmedian(np.diff(x, axis=1))), float(np.nanmedian(np.diff(y, axis=0))))
    return {"yield": 1800.0, "hardening": 9500.0, "height": 3600.0, "centre_x": float(np.ptp(x)), "centre_y": float(np.ptp(y)), "variance": float(np.hypot(np.ptp(x), np.ptp(y)) ** 2 - (3 * spacing) ** 2), "angle": float(np.pi)}


def _common_directions(experiment):
    scales = _bounds(experiment)
    with ORACLE_PARAMETERS.open() as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["requested_bases"]) == 5]
    configuration = rows[0]["configuration"]
    rows = [row for row in rows if row["configuration"] == configuration]
    return [Direction("yield_homogeneous", "homogeneous", "yield_strength", scales["yield"]), *[Direction(f"oracle_basis_{index}_height", "height", "yield_strength", scales["height"], _csv_kernel(row)) for index, row in enumerate(rows, start=1)], Direction("hardening_homogeneous", "homogeneous", "hardening_modulus", scales["hardening"])]


def _full_directions(control, experiment):
    scales = _bounds(experiment)
    result = [Direction("yield_homogeneous", "homogeneous", "yield_strength", scales["yield"]), Direction("hardening_homogeneous", "homogeneous", "hardening_modulus", scales["hardening"])]
    for index, kernel in enumerate(_kernels(control), start=1):
        result.append(Direction(f"basis_{index}_height", "height", "yield_strength", scales["height"], kernel))
        for attribute, scale in (("centre_x", scales["centre_x"]), ("centre_y", scales["centre_y"]), ("variance_x", scales["variance"]), ("variance_y", scales["variance"]), ("angle", scales["angle"])):
            result.append(Direction(f"basis_{index}_{attribute}", "geometry", "yield_strength", scale, kernel, attribute))
    return result


def _csv_kernel(row):
    return {"centre_x": float(row["centre_x_mm"]), "centre_y": float(row["centre_y_mm"]), "variance_x": float(row["sigma_major_mm"]) ** 2, "variance_y": float(row["sigma_minor_mm"]) ** 2, "angle": float(row["angle_radians"]), "height": float(row["height_mpa"])}


def _kernels(control):
    kernels = []
    for item in control.history.phases[1].final_snapshot.spatial_parameterisations["yield_strength"]:
        if item.summary.get("kind") == "basis_functions":
            for kernel in item.summary.get("kernels", []):
                kernels.append({"centre_x": float(kernel["centre"][0]), "centre_y": float(kernel["centre"][1]), "variance_x": float(kernel["variance"][0]), "variance_y": float(kernel["variance"][1]), "angle": float(kernel["angle"]), "height": float(kernel["height"])})
    return kernels


def _kernel(kernel, x, y):
    dx, dy = x - kernel["centre_x"], y - kernel["centre_y"]
    c, s = np.cos(kernel["angle"]), np.sin(kernel["angle"])
    return np.exp(-0.5 * ((c * dx + s * dy) ** 2 / kernel["variance_x"] + (-s * dx + c * dy) ** 2 / kernel["variance_y"]))


def _perturb(maps, direction, sign, h, experiment):
    result = {name: value.copy() for name, value in maps.items()}
    magnitude = sign * h * direction.span
    if direction.kind == "homogeneous":
        result[direction.target] += magnitude
        return result
    assert direction.kernel is not None
    x, y = experiment.specimen_geometry.x, experiment.specimen_geometry.y
    if direction.kind == "height":
        result["yield_strength"] += magnitude * _kernel(direction.kernel, x, y)
        return result
    changed = dict(direction.kernel)
    changed[direction.attribute] += magnitude
    if changed[direction.attribute] <= 0.0 and direction.attribute.startswith("variance"):
        changed[direction.attribute] = direction.kernel[direction.attribute]
    result["yield_strength"] += direction.kernel["height"] * (_kernel(changed, x, y) - _kernel(direction.kernel, x, y))
    return result


def _jacobian_suite(states, directions, experiment, control, metrics, baselines, cache, output):
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for state in states:
        labels = [item.label for item in directions]
        columns, production_columns, stability = [], [], []
        for direction in directions:
            estimates = []
            base = _evaluate_maps(state.name, state.maps, experiment, control, metrics, baselines, cache)
            for h in STEPS:
                tag = f"fd2_{state.name}_{direction.label}_{h:.0e}"
                plus = _evaluate_maps(tag + "_plus", _perturb(state.maps, direction, 1, h, experiment), experiment, control, metrics, baselines, cache)
                minus = _evaluate_maps(tag + "_minus", _perturb(state.maps, direction, -1, h, experiment), experiment, control, metrics, baselines, cache)
                one_sided = (
                    direction.kind == "geometry"
                    and direction.attribute.startswith("variance")
                    and np.array_equal(minus["residual"], base["residual"])
                )
                if one_sided:
                    estimates.append(((plus["residual"] - base["residual"]) / h, (plus["residual_production"] - base["residual_production"]) / h))
                else:
                    estimates.append(((plus["residual"] - minus["residual"]) / (2 * h), (plus["residual_production"] - minus["residual_production"]) / (2 * h)))
            columns.append(estimates[1][0]); production_columns.append(estimates[1][1])
            cosine = float(np.dot(estimates[0][0], estimates[2][0]) / max(np.linalg.norm(estimates[0][0]) * np.linalg.norm(estimates[2][0]), 1e-300))
            change = float(abs(np.linalg.norm(estimates[0][0]) - np.linalg.norm(estimates[2][0])) / max(np.linalg.norm(estimates[1][0]), 1e-300))
            stability.append({"label": direction.label, "cosine_h_small_large": cosine, "relative_norm_change": change, "stable": bool(cosine >= .995 and change <= .05)})
        jacobian, production = np.column_stack(columns), np.column_stack(production_columns)
        np.savez_compressed(output / f"{state.name}.npz", jacobian=jacobian, production_jacobian=production, labels=np.asarray(labels))
        summaries[state.name] = _svd_report(jacobian, production, labels, stability)
        _plot_svd(jacobian, labels, output / f"{state.name}.png", state.name)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))
    return summaries


def _svd_report(jacobian, production, labels, stability):
    def calculate(values):
        _, singular, right = np.linalg.svd(values, full_matrices=False)
        norms = np.linalg.norm(values, axis=0)
        correlation = values.T @ values / np.maximum(np.outer(norms, norms), 1e-300)
        return {"singular_values": singular.tolist(), "condition_number": float(singular[0] / max(singular[-1], 1e-300)), "ranks": {str(value): int(np.count_nonzero(singular / singular[0] >= value)) for value in (1e-2, 1e-3, 1e-4)}, "right_vectors": right.tolist(), "column_correlation": correlation.tolist()}
    return {"labels": labels, "stability": stability, "block_normalised": calculate(jacobian), "production_weighted": calculate(production)}


def _plot_svd(jacobian, labels, path, title):
    _, singular, _ = np.linalg.svd(jacobian, full_matrices=False)
    norms = np.linalg.norm(jacobian, axis=0); corr = jacobian.T @ jacobian / np.maximum(np.outer(norms, norms), 1e-300)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].semilogy(np.arange(1, len(singular) + 1), singular / singular[0], "o-"); axes[0].set(xlabel="Mode", ylabel=r"$\sigma_i/\sigma_1$", title=f"{title}: dimensionless SVD")
    image = axes[1].imshow(corr, vmin=-1, vmax=1, cmap="coolwarm"); axes[1].set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels, title="Column correlation"); axes[1].tick_params(axis="x", labelrotation=90, labelsize=6); axes[1].tick_params(axis="y", labelsize=6); figure.colorbar(image, ax=axes[1]); figure.savefig(path, dpi=180); plt.close(figure)


def _fe_closure(experiment, metrics, baselines, cache, output):
    output.mkdir(parents=True, exist_ok=True)
    stress, nearest_stress, metadata = _load_fe_stress(experiment)
    np.savez_compressed(output / "mapped_native_fe_stress.npz", stress=stress, nearest_stress=nearest_stress, **metadata)
    values = _evaluate_stress("native_fe_linear", stress, {}, experiment, metrics, baselines, cache)
    nearest = _evaluate_stress("native_fe_nearest", nearest_stress, {}, experiment, metrics, baselines, cache)
    summary = {"objective": float(values["objective"]), "egi_scalars": values["egi_scalars"].tolist(), "force_scalar": float(values["force_scalar"]), "nearest_objective": float(nearest["objective"]), "nearest_egi_scalars": nearest["egi_scalars"].tolist(), "nearest_force_scalar": float(nearest["force_scalar"]), **{key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in metadata.items()}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot_fe_stress(experiment, stress, output / "native_fe_vm_final.png")
    return summary


def _load_fe_stress(experiment):
    native = CASE / "fe-data/ansys-export"
    elements = np.genfromtxt(native / "elements.csv", delimiter=",", names=True)
    results = np.genfromtxt(native / "element_results.csv", delimiter=",", names=True)
    lookup = {int(row["element"]): (float(row["x_mm"]), float(row["y_mm"])) for row in elements}
    expected = np.arange(1, experiment.timesteps.size + 1)
    ids = np.unique(results["element"])
    if ids.size != elements.size or any(int(value) not in lookup for value in ids):
        raise ValueError("Native results/elements join is incomplete.")
    xy = np.asarray([lookup[int(value)] for value in ids])
    x, y = experiment.specimen_geometry.x, experiment.specimen_geometry.y
    mask = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(x, y)
    fields = np.full((experiment.timesteps.size, 3, *x.shape), np.nan)
    nearest_fields = np.full_like(fields, np.nan)
    fallback = []
    fe_time = []
    for time_index, set_value in enumerate(expected):
        subset = results[results["set"] == set_value]
        if subset.size != ids.size:
            raise ValueError(f"Incomplete native FE set {set_value}.")
        order = {int(value): index for index, value in enumerate(subset["element"])}
        values = np.column_stack([[float(subset[order[int(element)]][component]) for element in ids] for component in ("sig_xx_mpa", "sig_yy_mpa", "sig_xy_mpa")])
        interpolated = LinearNDInterpolator(xy, values, fill_value=np.nan)(x, y)
        missing = mask & np.any(~np.isfinite(interpolated), axis=2)
        fallback.append(int(np.count_nonzero(missing)))
        if np.any(missing):
            interpolated[missing] = NearestNDInterpolator(xy, values)(x[missing], y[missing])
        fields[time_index] = np.moveaxis(interpolated, -1, 0); fields[time_index, :, ~mask] = np.nan
        nearest_values = NearestNDInterpolator(xy, values)(x, y)
        nearest_fields[time_index] = np.moveaxis(nearest_values, -1, 0); nearest_fields[time_index, :, ~mask] = np.nan
        fe_time.append(float(subset["time"][0]))
    if not np.allclose(fe_time, experiment.timesteps, rtol=1e-10, atol=1e-12):
        raise ValueError("Native FE time values do not match prepared experiment time.")
    return fields, nearest_fields, {"fe_element_count": np.asarray(ids.size), "fe_timestep_count": np.asarray(len(expected)), "linear_fallback_count_by_timestep": np.asarray(fallback), "fe_time": np.asarray(fe_time)}


def _plot_fe_stress(experiment, stress, path):
    sx, sy, txy = stress[-1]
    vm = np.sqrt(sx*sx - sx*sy + sy*sy + 3*txy*txy)
    figure, axis = plt.subplots(figsize=(8, 3), layout="constrained")
    image = axis.pcolormesh(experiment.specimen_geometry.x, experiment.specimen_geometry.y, vm, shading="auto", cmap="viridis")
    figure.colorbar(image, ax=axis, label="von Mises stress [MPa]")
    axis.set(title="Mapped native FE stress: final timestep", xlabel="x [mm]", ylabel="y [mm]")
    figure.savefig(path, dpi=180); plt.close(figure)


def _write_closure(output, closure, control):
    rows = [{"state": name, "objective": float(item["objective"]), "egi_29": float(item["egi_scalars"][0]), "egi_57": float(item["egi_scalars"][1]), "fre": float(item["force_scalar"])} for name, item in closure.items()]
    with (output / "closure.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    accepted = [solve for solve in control.history.phases[1].solve_results if solve.accepted]
    expected = float(accepted[-1].final_objective["cost"]); measured = next(row["objective"] for row in rows if row["state"] == "identified_seed0")
    (output / "closure.json").write_text(json.dumps({"rows": rows, "saved_accepted_objective": expected, "recomposed_objective": measured, "relative_difference": abs(expected-measured)/expected}, indent=2))


def _write_report(output, closure, jacobians, fe):
    lines = ["# Notched EBW closure and dimensionless residual-Jacobian diagnostic", "", "## Closure", "", "| State | Objective | EGI-29 | EGI-57 | FRE |", "|---|---:|---:|---:|---:|"]
    for name, item in closure.items(): lines.append(f"| {name} | {float(item['objective']):.6g} | {float(item['egi_scalars'][0]):.4g} | {float(item['egi_scalars'][1]):.4g} | {float(item['force_scalar']):.4g} |")
    if fe: lines.extend(["", "## Native FE closure", "", f"- Linear-centroid objective: {fe['objective']:.6g}", f"- Linear EGI-29/57: {fe['egi_scalars'][0]:.4g}, {fe['egi_scalars'][1]:.4g}; FRE: {fe['force_scalar']:.4g}", f"- Nearest-centroid objective: {fe['nearest_objective']:.6g}", f"- Nearest EGI-29/57: {fe['nearest_egi_scalars'][0]:.4g}, {fe['nearest_egi_scalars'][1]:.4g}; FRE: {fe['nearest_force_scalar']:.4g}", f"- Linear interpolation fallback count by timestep: {fe['linear_fallback_count_by_timestep']}"])
    lines.extend(["", "## Dimensionless Jacobian SVD", ""])
    for suite, states in jacobians.items():
        lines.append(f"### {suite}")
        for state, item in states.items():
            block = item["block_normalised"]; stable = sum(row["stable"] for row in item["stability"])
            lines.append(f"- {state}: condition number {block['condition_number']:.3g}; ranks {block['ranks']}; stable finite-difference columns {stable}/{len(item['stability'])}.")
    (output / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
