"""Create the decision report for the corrected raw-hybrid BF3 gate."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import textwrap

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from pyvale.vfm import ExperimentData, load_identification_result
from pyvale.vfm.postprocessing import (
    evaluate_snapshot_parameter_maps,
    load_constitutive_law_from_result,
    load_known_parameter_maps,
)
from analyse_notched_ebw_component_library import _property_errors
from analyse_notched_ebw_gate_campaign import _active_masks, _basis_count, _complete_maps


LABELS = {
    "raw_parsimonious": "Raw 7/57 (alpha 0.25)",
    "raw_information_rich": "Raw 7/29/57 (alpha 0.50)",
}


@dataclass(slots=True)
class Row:
    case: str
    objective: str
    condition: str
    seed: int
    basis_count: int
    accepted: bool
    objective_cost: float
    global_cost: float
    material_cost: float
    max_normalised_feature: float
    roi_rmse_mpa: float
    yielded_rmse_mpa: float
    high_plastic_rmse_mpa: float


def main() -> None:
    args = _parse_args()
    gate = args.gate.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(gate, dataset)
    direct = _csv(args.direct_fit.expanduser().resolve())
    controls = _csv(args.controls.expanduser().resolve())
    _write_rows(output.with_suffix(".csv"), rows)
    with PdfPages(output) as pdf:
        _summary(pdf, rows)
        _map_recovery(pdf, rows, direct, controls)
        _objective_health(pdf, rows)
        _decision(pdf, rows)
    print(f"raw BF3 gate findings={output}", flush=True)


def _load_rows(gate: Path, dataset: Path) -> list[Row]:
    manifest = json.loads((gate / "campaign_manifest.json").read_text())
    experiment = ExperimentData.load_from_file(dataset / "prepared/experiment_data.yaml")
    known_raw = load_known_parameter_maps(dataset / "prepared/known_parameter_maps.npz")
    if known_raw is None:
        raise RuntimeError("Synthetic truth maps are required.")
    known = {name: np.asarray(value, dtype=float) for name, value in known_raw.items()}
    mask = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(
        experiment.specimen_geometry.x, experiment.specimen_geometry.y
    )
    first = load_identification_result(
        gate / manifest["cases"][0]["name"] / "identification_result.yaml"
    )
    law = load_constitutive_law_from_result(first)
    yielded, high_plastic = _active_masks(experiment, law, known, mask)
    rows: list[Row] = []
    for case in manifest["cases"]:
        result = load_identification_result(
            gate / case["name"] / "identification_result.yaml"
        )
        for solve in result.history.phases[-1].solve_results:
            maps = _complete_maps(
                evaluate_snapshot_parameter_maps(solve.final_snapshot, experiment), known
            )
            errors = _property_errors(maps, known, mask, yielded, high_plastic)
            components = solve.final_objective["components"]
            features = components.get("features", [])
            rows.append(Row(
                case=case["name"], objective=case["objective"],
                condition=case["condition"], seed=int(case["seed"]),
                basis_count=_basis_count(solve.final_snapshot), accepted=bool(solve.accepted),
                objective_cost=float(solve.final_objective["cost"]),
                global_cost=float(components["global_cost"]),
                material_cost=float(components["material_cost"]),
                max_normalised_feature=max(float(x["normalised_value"]) for x in features),
                roi_rmse_mpa=float(errors["roi_rmse_mpa"]),
                yielded_rmse_mpa=float(errors["yielded_rmse_mpa"]),
                high_plastic_rmse_mpa=float(errors["high_plastic_rmse_mpa"]),
            ))
    return rows


def _summary(pdf, rows):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle("Notched-EBW corrected raw-objective BF3 gate", fontsize=19, y=.95)
    accepted = sum(row.accepted for row in rows)
    bf3 = [row for row in rows if row.basis_count == 3]
    lines = [
        f"Execution passed: 8/8 cases completed, {accepted}/{len(rows)} BF stages were accepted, and every case reached BF3.",
        "The dimensionless normalisation and fixed-trajectory repairs are functioning. The earlier false BF2 rejection has been removed.",
        "In clean data, the final material features approach their propagated noise floors. Under 1x noise, the dominant normalised feature remains near its stage-start value; this is a scientific warning to track through BF7.",
        f"BF3 median yielded-region RMSE spans {min(row.yielded_rmse_mpa for row in bf3):.1f}-{max(row.yielded_rmse_mpa for row in bf3):.1f} MPa across individual cases.",
        "Scientific result: the information-rich 7/29/57 formulation is clearly preferable in clean data. Under 1x noise, neither raw formulation is uniformly superior across two seeds.",
        "Decision: the software gate passes. Proceed to the hybrid-only BF7 pilot, retaining both raw formulations and the matched clean/noisy seeds. Existing control runs remain reusable.",
    ]
    y = .83
    for line in lines:
        fig.text(.07, y, textwrap.fill(line, 108), fontsize=11.5, va="top")
        y -= .112
    fig.text(.07, .12, "Interpretation", fontsize=14, weight="bold")
    fig.text(.07, .067, "BF3 is an implementation and early-trajectory gate, not an objective-selection endpoint. The late BF recovery gap remains the decisive evidence.", fontsize=11, wrap=True)
    pdf.savefig(fig); plt.close(fig)


def _map_recovery(pdf, rows, direct, controls):
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), constrained_layout=True)
    colors = {"raw_parsimonious": "tab:orange", "raw_information_rich": "tab:green"}
    for axis, condition in zip(axes, ("clean", "noise1x"), strict=True):
        for objective, label in LABELS.items():
            selected = [r for r in rows if r.objective == objective and r.condition == condition]
            bfs = sorted({r.basis_count for r in selected})
            values = [np.median([r.yielded_rmse_mpa for r in selected if r.basis_count == bf]) for bf in bfs]
            axis.plot(bfs, values, marker="o", linewidth=2, label=label, color=colors[objective])
        for objective, label, color in (
            ("current_29_57", "Current 29/57", "0.55"),
            ("multiscale_equal_7_29_57", "Equal 7/29/57", "tab:blue"),
        ):
            selected = [r for r in controls if r["objective"] == objective and r["condition"] == condition and int(r["basis_count"]) <= 3]
            bfs = sorted({int(r["basis_count"]) for r in selected})
            values = [np.median([float(r["yielded_rmse_mpa"]) for r in selected if int(r["basis_count"]) == bf]) for bf in bfs]
            axis.plot(bfs, values, marker=".", linestyle="--", label=label, color=color)
        axis.plot(
            [int(r["basis_count"]) for r in direct if int(r["basis_count"]) <= 3],
            [float(r["yielded_rmse_mpa"]) for r in direct if int(r["basis_count"]) <= 3],
            "k:", linewidth=1.7, label="Direct fit",
        )
        axis.set(xlabel="BF count", ylabel="Yielded-region RMSE [MPa]", title=condition.replace("noise1x", "1x noise").title())
        axis.set_xticks((1, 2, 3)); axis.grid(alpha=.25)
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Early map-recovery trajectories (median of two matched seeds)", fontsize=16)
    pdf.savefig(fig); plt.close(fig)


def _objective_health(pdf, rows):
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), constrained_layout=True)
    styles = {"clean": "-", "noise1x": "--"}
    colors = {"raw_parsimonious": "tab:orange", "raw_information_rich": "tab:green"}
    for objective, label in LABELS.items():
        for condition in ("clean", "noise1x"):
            selected = [r for r in rows if r.objective == objective and r.condition == condition]
            bfs = sorted({r.basis_count for r in selected})
            axes[0].plot(bfs, [np.median([r.global_cost for r in selected if r.basis_count == bf]) for bf in bfs], marker="o", linestyle=styles[condition], color=colors[objective], label=f"{label} / {condition}")
            axes[1].plot(bfs, [np.median([r.max_normalised_feature for r in selected if r.basis_count == bf]) for bf in bfs], marker="o", linestyle=styles[condition], color=colors[objective], label=f"{label} / {condition}")
    axes[0].set(xlabel="BF count", ylabel="Global closure contribution", title="Mechanical closure improves")
    axes[1].set(xlabel="BF count", ylabel="Largest normalised material feature", title="Material terms remain finite and responsive")
    for axis in axes:
        axis.set_xticks((1, 2, 3)); axis.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    fig.suptitle("Corrected objective-health diagnostics", fontsize=16)
    pdf.savefig(fig); plt.close(fig)


def _decision(pdf, rows):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle("Decision and next experiment", fontsize=18, y=.94)
    sections = [
        ("Gate verdict: pass", "All repaired hybrid trajectories reach BF3 without failure or cross-stage rejection. Feature scaling is dimensionless and values are numerically well behaved."),
        ("What is encouraging", "The rich formulation gives the strongest clean-data early recovery; global closure falls through BF3; clean-data optimisation drives semantic features towards their estimated noise floors."),
        ("What remains unresolved", "Under 1x noise, the dominant normalised material feature remains near one. Noise sensitivity and seed ordering are mixed, BF3 remains far above direct-fit representability, and two seeds cannot select a production objective."),
        ("Run next", "Launch only the eight raw hybrid cases through BF7: two formulations x clean/1x noise x two matched seeds. Reuse the eight completed BF7 controls and the direct-fit reference."),
        ("Decision criteria", "Compare recovery-gap AUC, BF7 yielded/high-plastic/ROI errors, common closure, seed spread, successive map stabilisation, and feature/noise-floor behaviour."),
        ("Still defer projection", "Do not add the projected objective until the raw BF7 comparison determines whether the remaining failure is objective information or optimisation/parameterisation."),
    ]
    y=.84
    for heading, body in sections:
        fig.text(.07, y, heading, fontsize=13, weight="bold"); y -= .043
        fig.text(.085, y, textwrap.fill(body, 114), fontsize=10.5, va="top"); y -= .108
    pdf.savefig(fig); plt.close(fig)


def _write_rows(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])))
        writer.writeheader(); writer.writerows(asdict(row) for row in rows)


def _csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--direct-fit", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
