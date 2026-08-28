"""Analyse every solve in a notched-EBW gate campaign.

All states are recomputed against one common phase-0 baseline.  The report
compares the production mechanical objective with a sensitivity-active
diagnostic whose residual-row weights are calculated at each candidate state.
Synthetic truth is used only for validation and for controlled perturbations;
it is not used to construct the sensitivity-active score.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from pyvale.vfm import (
    EquilibriumGapMetric,
    ExperimentData,
    SliceConfig,
    SliceWiseForceReconstructionMetric,
    load_identification_result,
)
from pyvale.vfm.postprocessing import (
    compute_plasticity_diagnostics,
    evaluate_snapshot_parameter_maps,
    load_constitutive_law_from_result,
    load_known_parameter_maps,
)


WINDOWS = (29, 57)
WINDOW_WEIGHTS = np.asarray(WINDOWS, dtype=np.float64) / sum(WINDOWS)
FORCE_WEIGHT = 0.1
YIELD_BOUNDS = (200.0, 2000.0)
HARDENING_BOUNDS = (500.0, 10_000.0)
SENSITIVITY_STEP = 1.0e-3


@dataclass(slots=True)
class StateMetrics:
    state_id: str
    source: str
    case_name: str
    policy: str
    gate: float | None
    seed: int | None
    solve_index: int | None
    basis_count: int | None
    accepted: bool | None
    is_final_accepted: bool
    is_best_visited: bool
    objective: float
    active_objective: float
    egi_29: float
    egi_57: float
    fre: float
    roi_rmse_mpa: float
    yielded_rmse_mpa: float
    high_plastic_rmse_mpa: float
    yielded_mape_percent: float
    yielded_above_5pct: float
    yielded_above_10pct: float
    yielded_above_15pct: float
    hardening_error_percent: float


def main() -> None:
    args = _parse_args()
    campaign_root = args.campaign_root.expanduser().resolve()
    manifest = json.loads(
        (campaign_root / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    dataset = args.dataset.expanduser().resolve() if args.dataset else Path(manifest["dataset"])
    prepared = dataset / "prepared"
    output = args.output.expanduser().resolve() if args.output else campaign_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    experiment = ExperimentData.load_from_file(prepared / "experiment_data.yaml")
    known = load_known_parameter_maps(prepared / "known_parameter_maps.npz")
    if known is None:
        raise RuntimeError("Known parameter maps are required for this synthetic campaign.")
    known = {name: np.asarray(value, dtype=np.float64) for name, value in known.items()}
    mask = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(
        experiment.specimen_geometry.x, experiment.specimen_geometry.y
    )

    run_items = _completed_runs(campaign_root, manifest)
    if not run_items:
        raise RuntimeError(f"No completed campaign runs found below {campaign_root}.")
    first_result = load_identification_result(run_items[0][1])
    law = load_constitutive_law_from_result(first_result)
    metrics = _create_metrics(experiment)
    baselines = _extract_baselines(first_result)
    yielded, high_plastic = _active_masks(experiment, law, known, mask)

    states: list[StateMetrics] = []
    for metadata, result_path in run_items:
        result = load_identification_result(result_path)
        solve_rows: list[StateMetrics] = []
        for solve in result.history.phases[-1].solve_results:
            if solve.final_snapshot is None:
                continue
            maps = _complete_maps(
                evaluate_snapshot_parameter_maps(solve.final_snapshot, experiment),
                known,
            )
            basis_count = _basis_count(solve.final_snapshot)
            solve_rows.append(
                _evaluate_state(
                    state_id=f"{metadata['name']}/solve_{solve.solve_iteration}",
                    source="campaign",
                    case_name=metadata["name"],
                    policy=metadata["policy"],
                    gate=float(metadata["gate"]),
                    seed=int(metadata["seed"]),
                    solve_index=int(solve.solve_iteration),
                    basis_count=basis_count,
                    accepted=bool(solve.accepted),
                    maps=maps,
                    known=known,
                    experiment=experiment,
                    law=law,
                    metrics=metrics,
                    baselines=baselines,
                    mask=mask,
                    yielded=yielded,
                    high_plastic=high_plastic,
                )
            )
        accepted_rows = [row for row in solve_rows if row.accepted]
        if accepted_rows:
            accepted_rows[-1].is_final_accepted = True
        if solve_rows:
            min(solve_rows, key=lambda row: row.objective).is_best_visited = True
        states.extend(solve_rows)

    for label, maps in _controlled_states(known, experiment, mask):
        states.append(
            _evaluate_state(
                state_id=f"controlled/{label}",
                source="controlled",
                case_name=label,
                policy="controlled",
                gate=None,
                seed=None,
                solve_index=None,
                basis_count=None,
                accepted=None,
                maps=maps,
                known=known,
                experiment=experiment,
                law=law,
                metrics=metrics,
                baselines=baselines,
                mask=mask,
                yielded=yielded,
                high_plastic=high_plastic,
            )
        )

    _write_states(output / "state_metrics.csv", states)
    gate_rows = _gate_summary(states)
    _write_dict_csv(output / "gate_summary.csv", gate_rows)
    ranks = _rank_summary(states)
    (output / "rank_summary.json").write_text(
        json.dumps(ranks, indent=2), encoding="utf-8"
    )
    _plot_objective_discrimination(output, states)
    _plot_gate_summary(output, states)
    _write_report(output / "REPORT.md", manifest, states, gate_rows, ranks)
    print(json.dumps({
        "output": str(output),
        "campaign_states": sum(row.source == "campaign" for row in states),
        "controlled_states": sum(row.source == "controlled" for row in states),
        "report": str(output / "REPORT.md"),
    }, indent=2))


def _completed_runs(
    campaign_root: Path, manifest: dict[str, object]
) -> list[tuple[dict[str, object], Path]]:
    runs = []
    for item in manifest["cases"]:
        result = campaign_root / str(item["name"]) / "identification_result.yaml"
        if result.is_file():
            runs.append((item, result))
    return runs


def _create_metrics(experiment: ExperimentData):
    egi = [EquilibriumGapMetric(window_size=(size, size)) for size in WINDOWS]
    force = SliceWiseForceReconstructionMetric(
        slice_config=SliceConfig(axis="x", num_slices=63)
    )
    for metric in [force, *egi]:
        metric.initialise(experiment)
    return force, egi


def _extract_baselines(result) -> tuple[np.ndarray, float]:
    accepted = [
        solve for solve in result.history.phases[-1].solve_results if solve.accepted
    ]
    if not accepted:
        raise RuntimeError("Reference run has no accepted phase-1 solve.")
    components = accepted[-1].final_objective["components"]
    return (
        np.asarray(components["egi_baselines"], dtype=np.float64),
        float(components["force_baseline"]),
    )


def _active_masks(experiment, law, known, mask):
    plasticity = compute_plasticity_diagnostics(experiment, law, known)
    if plasticity is None:
        raise RuntimeError("Plasticity diagnostics are unavailable.")
    yielded = np.asarray(plasticity.yielded_datapoints, dtype=bool) & mask
    peak = np.nanmax(
        np.asarray(plasticity.equivalent_plastic_strain, dtype=np.float64), axis=0
    )
    threshold = float(np.nanpercentile(peak[yielded], 75.0))
    return yielded, yielded & (peak >= threshold)


def _complete_maps(partial, known):
    maps = {name: value.copy() for name, value in known.items()}
    maps.update({name: np.asarray(value, dtype=np.float64) for name, value in partial.items()})
    maps["yield_strength"] = np.clip(maps["yield_strength"], *YIELD_BOUNDS)
    maps["hardening_modulus"] = np.clip(maps["hardening_modulus"], *HARDENING_BOUNDS)
    return maps


def _basis_count(snapshot) -> int:
    for item in snapshot.spatial_parameterisations.get("yield_strength", []):
        if item.summary.get("kind") == "basis_functions":
            return int(item.summary.get("num_kernels", len(item.summary.get("kernels", []))))
    return 0


def _residual_blocks(law, maps, experiment, metrics, baselines):
    stress = law.calculate_stress(experiment.strain, maps)
    force, egi = metrics
    blocks = []
    scalars = []
    for index, metric in enumerate(egi):
        result = metric.evaluate_equilibrium_gap(stress)
        values = np.asarray(result.normalised_gap, dtype=np.float64)
        temporal = np.asarray(
            result.metric_result.additional_fields["temporal_weights"],
            dtype=np.float64,
        )
        valid = np.isfinite(values)
        block = values * np.sqrt(temporal)[:, None, None]
        block /= np.sqrt(np.count_nonzero(valid)) * baselines[0][index]
        block[~valid] = np.nan
        blocks.append(block)
        scalars.append(float(np.sqrt(np.nansum(block**2))))
    force_result = force.evaluate_force_recon_error(stress, experiment)
    metadata = force_result.metric_result.additional_fields
    block = np.asarray(metadata["normalised_residual"], dtype=np.float64)
    block = block * np.sqrt(np.asarray(metadata["temporal_weights"], dtype=float))[:, None]
    block = block * np.sqrt(np.asarray(metadata["spatial_weights"], dtype=float))[None, :]
    block /= baselines[1]
    blocks.append(block)
    scalars.append(float(np.sqrt(np.nansum(block**2))))
    return blocks, np.asarray(scalars, dtype=np.float64)


def _objective_from_scalars(scalars: np.ndarray) -> float:
    coefficients = np.asarray([
        (1.0 - FORCE_WEIGHT) * WINDOW_WEIGHTS[0],
        (1.0 - FORCE_WEIGHT) * WINDOW_WEIGHTS[1],
        FORCE_WEIGHT,
    ])
    return float(np.dot(coefficients, scalars))


def _sensitivity_active_objective(
    law, maps, experiment, metrics, baselines, base_blocks
) -> float:
    span = YIELD_BOUNDS[1] - YIELD_BOUNDS[0]
    plus = {name: value.copy() for name, value in maps.items()}
    minus = {name: value.copy() for name, value in maps.items()}
    plus["yield_strength"] = np.clip(
        plus["yield_strength"] + SENSITIVITY_STEP * span, *YIELD_BOUNDS
    )
    minus["yield_strength"] = np.clip(
        minus["yield_strength"] - SENSITIVITY_STEP * span, *YIELD_BOUNDS
    )
    upper, _ = _residual_blocks(law, plus, experiment, metrics, baselines)
    lower, _ = _residual_blocks(law, minus, experiment, metrics, baselines)
    weighted_scalars = []
    for residual, high, low in zip(base_blocks, upper, lower, strict=True):
        sensitivity = np.abs((high - low) / (2.0 * SENSITIVITY_STEP))
        valid = np.isfinite(residual) & np.isfinite(sensitivity)
        if not np.any(valid):
            weighted_scalars.append(float("nan"))
            continue
        cap = float(np.nanpercentile(sensitivity[valid], 95.0))
        if cap <= 0.0:
            weight = np.ones_like(sensitivity)
        else:
            activity = np.clip(sensitivity / cap, 0.0, 1.0)
            weight = 0.05 + 0.95 * activity
            weight /= np.mean(weight[valid])
        weighted_scalars.append(float(np.sqrt(np.nansum(weight * residual**2))))
    return _objective_from_scalars(np.asarray(weighted_scalars))


def _evaluate_state(
    *, state_id, source, case_name, policy, gate, seed, solve_index,
    basis_count, accepted, maps, known, experiment, law, metrics, baselines,
    mask, yielded, high_plastic,
) -> StateMetrics:
    blocks, scalars = _residual_blocks(law, maps, experiment, metrics, baselines)
    active_objective = _sensitivity_active_objective(
        law, maps, experiment, metrics, baselines, blocks
    )
    error = maps["yield_strength"] - known["yield_strength"]
    relative = np.abs(error / known["yield_strength"])
    hardening = float(np.nanmean(maps["hardening_modulus"][mask]))
    true_hardening = float(np.nanmean(known["hardening_modulus"][mask]))
    return StateMetrics(
        state_id=state_id, source=source, case_name=case_name, policy=policy,
        gate=gate, seed=seed, solve_index=solve_index, basis_count=basis_count,
        accepted=accepted, is_final_accepted=False, is_best_visited=False,
        objective=_objective_from_scalars(scalars),
        active_objective=active_objective,
        egi_29=float(scalars[0]), egi_57=float(scalars[1]), fre=float(scalars[2]),
        roi_rmse_mpa=_rmse(error, mask),
        yielded_rmse_mpa=_rmse(error, yielded),
        high_plastic_rmse_mpa=_rmse(error, high_plastic),
        yielded_mape_percent=100.0 * float(np.mean(relative[yielded])),
        yielded_above_5pct=float(np.mean(relative[yielded] > 0.05)),
        yielded_above_10pct=float(np.mean(relative[yielded] > 0.10)),
        yielded_above_15pct=float(np.mean(relative[yielded] > 0.15)),
        hardening_error_percent=100.0 * (hardening - true_hardening) / true_hardening,
    )


def _controlled_states(known, experiment, mask):
    yield_truth = known["yield_strength"]
    yield "truth", {name: value.copy() for name, value in known.items()}
    for offset in (-100.0, -50.0, -20.0, 20.0, 50.0, 100.0):
        maps = {name: value.copy() for name, value in known.items()}
        maps["yield_strength"] = np.clip(yield_truth + offset, *YIELD_BOUNDS)
        yield f"uniform_{offset:+.0f}mpa", maps

    x = np.asarray(experiment.specimen_geometry.x, dtype=np.float64)
    y = np.asarray(experiment.specimen_geometry.y, dtype=np.float64)
    x_values = x[mask]
    y_values = y[mask]
    for centre_fraction in (0.25, 0.5, 0.75):
        centre_x = float(np.quantile(x_values, centre_fraction))
        centre_y = float(np.median(y_values))
        for scale_fraction in (0.05, 0.15):
            scale_x = scale_fraction * float(np.ptp(x_values))
            scale_y = scale_fraction * float(np.ptp(y_values))
            feature = np.exp(-0.5 * (
                ((x - centre_x) / scale_x) ** 2
                + ((y - centre_y) / scale_y) ** 2
            ))
            for amplitude in (-75.0, -25.0, 25.0, 75.0):
                maps = {name: value.copy() for name, value in known.items()}
                maps["yield_strength"] = np.clip(
                    yield_truth + amplitude * feature, *YIELD_BOUNDS
                )
                label = (
                    f"gaussian_x{centre_fraction:.2f}_s{scale_fraction:.2f}_"
                    f"a{amplitude:+.0f}"
                )
                yield label, maps


def _rmse(error, selection) -> float:
    return float(np.sqrt(np.mean(np.asarray(error)[selection] ** 2)))


def _gate_summary(states: list[StateMetrics]) -> list[dict[str, object]]:
    selected = [row for row in states if row.source == "campaign" and row.is_final_accepted]
    groups: dict[tuple[str, float], list[StateMetrics]] = {}
    for row in selected:
        assert row.gate is not None
        groups.setdefault((row.policy, row.gate), []).append(row)
    output = []
    for (policy, gate), rows in sorted(groups.items()):
        output.append({
            "policy": policy,
            "gate": gate,
            "runs": len(rows),
            "median_objective": float(np.median([row.objective for row in rows])),
            "median_active_objective": float(np.median([row.active_objective for row in rows])),
            "median_roi_rmse_mpa": float(np.median([row.roi_rmse_mpa for row in rows])),
            "median_yielded_rmse_mpa": float(np.median([row.yielded_rmse_mpa for row in rows])),
            "iqr_yielded_rmse_mpa": _iqr([row.yielded_rmse_mpa for row in rows]),
            "median_yielded_above_10pct": float(np.median([row.yielded_above_10pct for row in rows])),
            "median_basis_count": float(np.median([row.basis_count for row in rows])),
        })
    return output


def _iqr(values: Iterable[float]) -> float:
    lower, upper = np.percentile(list(values), [25.0, 75.0])
    return float(upper - lower)


def _rank_summary(states: list[StateMetrics]) -> dict[str, object]:
    result: dict[str, object] = {}
    for subset, rows in {
        "all": states,
        "campaign_all_solves": [row for row in states if row.source == "campaign"],
        "campaign_final_accepted": [row for row in states if row.is_final_accepted],
        "controlled": [row for row in states if row.source == "controlled"],
    }.items():
        result[subset] = {}
        for objective in ("objective", "active_objective"):
            result[subset][objective] = {}
            for error in (
                "roi_rmse_mpa", "yielded_rmse_mpa", "high_plastic_rmse_mpa",
                "yielded_above_10pct", "yielded_above_15pct",
            ):
                coefficient, p_value = spearmanr(
                    [getattr(row, objective) for row in rows],
                    [getattr(row, error) for row in rows],
                )
                result[subset][objective][error] = {
                    "spearman_r": float(coefficient),
                    "p_value": float(p_value),
                    "states": len(rows),
                }
    return result


def _write_states(path: Path, states: list[StateMetrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(states[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in states)


def _write_dict_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_objective_discrimination(output: Path, states: list[StateMetrics]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    colours = {"campaign": "tab:blue", "controlled": "tab:orange"}
    for source in colours:
        rows = [row for row in states if row.source == source]
        for axis, objective, title in zip(
            axes,
            ("objective", "active_objective"),
            ("Mechanical objective", "Sensitivity-active diagnostic"),
            strict=True,
        ):
            axis.scatter(
                [getattr(row, objective) for row in rows],
                [row.yielded_rmse_mpa for row in rows],
                s=15, alpha=0.65, label=source, color=colours[source],
            )
            axis.set(xlabel=title, ylabel="Yielded RMSE [MPa]")
    axes[0].legend()
    figure.savefig(output / "objective_vs_yielded_rmse.png", dpi=180)
    plt.close(figure)


def _plot_gate_summary(output: Path, states: list[StateMetrics]) -> None:
    rows = [
        row for row in states
        if row.is_final_accepted and row.policy == "sensitivity_correction"
    ]
    gates = sorted({row.gate for row in rows if row.gate is not None})
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    axes[0].boxplot(
        [[row.yielded_rmse_mpa for row in rows if row.gate == gate] for gate in gates],
        tick_labels=[f"{100 * gate:g}%" for gate in gates],
    )
    axes[0].set(xlabel="Acceptance gate", ylabel="Yielded RMSE [MPa]")
    axes[1].boxplot(
        [[100 * row.yielded_above_10pct for row in rows if row.gate == gate] for gate in gates],
        tick_labels=[f"{100 * gate:g}%" for gate in gates],
    )
    axes[1].set(xlabel="Acceptance gate", ylabel="Yielded points above 10% error [%]")
    figure.savefig(output / "gate_comparison.png", dpi=180)
    plt.close(figure)


def _write_report(path, manifest, states, gate_rows, ranks) -> None:
    final_rows = [row for row in states if row.is_final_accepted]
    best_gate = min(
        (row for row in gate_rows if row["policy"] == "sensitivity_correction"),
        key=lambda row: (row["median_yielded_rmse_mpa"], row["iqr_yielded_rmse_mpa"]),
    )
    current_rank = ranks["campaign_all_solves"]["objective"]["yielded_rmse_mpa"]["spearman_r"]
    active_rank = ranks["campaign_all_solves"]["active_objective"]["yielded_rmse_mpa"]["spearman_r"]
    lines = [
        "# Notched-EBW gate and objective-discrimination campaign",
        "",
        f"Generated: {datetime.now().astimezone().isoformat()}",
        "",
        f"Campaign: `{manifest['campaign_name']}`; final accepted runs: {len(final_rows)}.",
        "",
        "## Gate summary",
        "",
        "| Policy | Gate | Runs | Median J | Median yielded RMSE | IQR | Median >10% | Median BFs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gate_rows:
        lines.append(
            f"| {row['policy']} | {100 * row['gate']:.1f}% | {row['runs']} | "
            f"{row['median_objective']:.5f} | {row['median_yielded_rmse_mpa']:.2f} MPa | "
            f"{row['iqr_yielded_rmse_mpa']:.2f} MPa | "
            f"{100 * row['median_yielded_above_10pct']:.1f}% | {row['median_basis_count']:.1f} |"
        )
    lines.extend([
        "",
        "## Objective discrimination",
        "",
        f"Across all campaign solves, Spearman correlation with yielded RMSE is "
        f"{current_rank:.3f} for the mechanical objective and {active_rank:.3f} "
        "for the sensitivity-active diagnostic. Higher positive correlation means "
        "the score ranks material-map error more consistently.",
        "",
        "## Automated screening result",
        "",
        f"The lowest median yielded RMSE (IQR tie-break) occurred at the "
        f"{100 * best_gate['gate']:.1f}% sensitivity-growth gate. This is a "
        "synthetic screening result, not by itself a production acceptance rule.",
        "",
        "See `state_metrics.csv`, `gate_summary.csv`, `rank_summary.json`, and the PNG figures for the full evidence.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
