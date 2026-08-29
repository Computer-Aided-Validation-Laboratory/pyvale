"""Create a concise decision report for the notched-EBW raw hybrid pilot."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
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
from analyse_notched_ebw_gate_campaign import (
    _active_masks,
    _basis_count,
    _complete_maps,
)


LABELS = {
    "current_29_57": "Current 29/57",
    "multiscale_equal_7_29_57": "Equal 7/29/57",
    "raw_parsimonious": "Raw 7/57",
    "raw_information_rich": "Raw 7/29/57",
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
    global_cost: float | None
    roi_rmse_mpa: float
    yielded_rmse_mpa: float
    high_plastic_rmse_mpa: float


def main() -> None:
    args = _parse_args()
    pilot = args.pilot.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    direct_path = args.direct_fit.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(pilot, dataset)
    direct = _csv(direct_path)
    _write_rows(output.with_suffix(".csv"), rows)
    with PdfPages(output) as pdf:
        _summary(pdf, rows)
        _trajectories(pdf, rows, direct)
        _final_controls(pdf, rows, direct)
        _diagnosis_and_next(pdf, rows)
    print(f"raw pilot findings={output}", flush=True)


def _load_rows(pilot: Path, dataset: Path) -> list[Row]:
    manifest = json.loads((pilot / "campaign_manifest.json").read_text())
    experiment = ExperimentData.load_from_file(dataset / "prepared/experiment_data.yaml")
    known_raw = load_known_parameter_maps(dataset / "prepared/known_parameter_maps.npz")
    if known_raw is None:
        raise RuntimeError("Synthetic truth maps are required.")
    known = {name: np.asarray(value, dtype=float) for name, value in known_raw.items()}
    mask = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(
        experiment.specimen_geometry.x, experiment.specimen_geometry.y
    )
    first = load_identification_result(
        pilot / manifest["cases"][0]["name"] / "identification_result.yaml"
    )
    law = load_constitutive_law_from_result(first)
    yielded, high_plastic = _active_masks(experiment, law, known, mask)
    rows: list[Row] = []
    for case in manifest["cases"]:
        path = pilot / case["name"] / "identification_result.yaml"
        result = load_identification_result(path)
        for solve in result.history.phases[-1].solve_results:
            maps = _complete_maps(
                evaluate_snapshot_parameter_maps(solve.final_snapshot, experiment),
                known,
            )
            errors = _property_errors(maps, known, mask, yielded, high_plastic)
            components = solve.final_objective.get("components", {})
            rows.append(Row(
                case=case["name"], objective=case["objective"],
                condition=case["condition"], seed=int(case["seed"]),
                basis_count=_basis_count(solve.final_snapshot),
                accepted=bool(solve.accepted),
                objective_cost=float(solve.final_objective["cost"]),
                global_cost=(None if "global_cost" not in components else float(components["global_cost"])),
                roi_rmse_mpa=float(errors["roi_rmse_mpa"]),
                yielded_rmse_mpa=float(errors["yielded_rmse_mpa"]),
                high_plastic_rmse_mpa=float(errors["high_plastic_rmse_mpa"]),
            ))
    return rows


def _summary(pdf, rows):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle("Notched-EBW raw-objective pilot: decision", fontsize=19, y=.95)
    counts = {
        objective: max(row.basis_count for row in rows if row.objective == objective and row.accepted)
        for objective in LABELS
    }
    lines = [
        "Decision: do not rank or confirm the hybrid objectives from this run. The controls are valid, but both hybrids stopped at BF2 and restored BF1 in every clean/noisy seed.",
        f"Accepted trajectory lengths: current 29/57 BF{counts['current_29_57']}; equal 7/29/57 BF{counts['multiscale_equal_7_29_57']}; raw 7/57 BF{counts['raw_parsimonious']}; raw 7/29/57 BF{counts['raw_information_rich']}.",
        "The rejected BF2 maps often improved truth error substantially, showing that rejection was not evidence of lost identifiability.",
        "Cause 1: fixed-BF7 was not actually forced. A zero improvement threshold still rejects a stage when its scalar rises.",
        "Cause 2: each BF solve refreshes feature references, so stage-normalised hybrid costs are not comparable across BF counts.",
        "Cause 3: the smooth-positive temperature was applied before normalisation. Its 0.001 raw-unit offset dominates EGI features of order 1e-5 to 1e-4, producing stage values around 5-20 rather than one.",
    ]
    y=.84
    for line in lines:
        fig.text(.07, y, textwrap.fill(line, 108), fontsize=11.2, va="top")
        y -= .108
    fig.text(.07, .105, "Scientific status", fontsize=14, weight="bold")
    fig.text(.07, .055, "Round 1 still identifies plausible feature families. This online run diagnoses objective lifecycle/scaling faults; it does not falsify the hybrid concept.", fontsize=11, wrap=True)
    pdf.savefig(fig); plt.close(fig)


def _trajectories(pdf, rows, direct):
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), constrained_layout=True)
    colors = dict(zip(LABELS, ("tab:gray", "tab:blue", "tab:orange", "tab:green"), strict=True))
    for axis, condition in zip(axes, ("clean", "noise1x"), strict=True):
        for objective, label in LABELS.items():
            selected = [row for row in rows if row.objective == objective and row.condition == condition]
            bfs = sorted({row.basis_count for row in selected})
            median = [np.median([row.yielded_rmse_mpa for row in selected if row.basis_count == bf]) for bf in bfs]
            accepted = [all(row.accepted for row in selected if row.basis_count == bf) for bf in bfs]
            axis.plot(bfs, median, marker="o", label=label, color=colors[objective])
            for bf, value, is_accepted in zip(bfs, median, accepted, strict=True):
                if not is_accepted:
                    axis.scatter([bf], [value], s=95, facecolors="none", edgecolors=colors[objective], linewidth=2)
        axis.plot(
            [int(row["basis_count"]) for row in direct],
            [float(row["yielded_rmse_mpa"]) for row in direct],
            "k--", linewidth=1.4, label="Direct fit" if condition == "clean" else None,
        )
        axis.set(xlabel="BF count", ylabel="Yielded-region RMSE [MPa]", title=condition.replace("noise1x", "1x noise").title())
        axis.set_xticks(range(0, 8)); axis.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Median trajectories; open markers are rejected hybrid BF2 states", fontsize=16)
    pdf.savefig(fig); plt.close(fig)


def _final_controls(pdf, rows, direct):
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), constrained_layout=True)
    control_names = ("current_29_57", "multiscale_equal_7_29_57")
    metrics = (("yielded_rmse_mpa", "Yielded"), ("high_plastic_rmse_mpa", "High plastic"), ("roi_rmse_mpa", "ROI"))
    x=np.arange(len(metrics)); width=.19
    combinations=[]
    for condition in ("clean", "noise1x"):
        for objective in control_names:
            selected=[row for row in rows if row.objective==objective and row.condition==condition and row.accepted and row.basis_count==7]
            combinations.append((condition, objective, [np.median([getattr(row,key) for row in selected]) for key,_ in metrics]))
    for index,(condition,objective,values) in enumerate(combinations):
        axes[0].bar(x+(index-1.5)*width,values,width,label=f"{LABELS[objective]} / {condition}")
    axes[0].set_xticks(x,[label for _,label in metrics]); axes[0].set(ylabel="BF7 RMSE [MPa]",title="Valid control endpoints")
    axes[0].grid(axis="y",alpha=.25); axes[0].legend(fontsize=7)
    bf=[int(row["basis_count"]) for row in direct]
    for key,label in metrics:
        axes[1].plot(bf,[float(row[key]) for row in direct],marker="o",label=label)
    axes[1].set(xlabel="BF count",ylabel="Direct-fit RMSE [MPa]",title="Representability reference")
    axes[1].grid(alpha=.25); axes[1].legend()
    fig.suptitle("Controls improve with BF count but remain far above direct fitting",fontsize=16)
    pdf.savefig(fig); plt.close(fig)


def _diagnosis_and_next(pdf, rows):
    fig = plt.figure(figsize=(11.69, 8.27)); fig.suptitle("Required repair and economical rerun", fontsize=18, y=.94)
    sections = [
        ("1. Make normalisation dimensionless", "Compute z=(feature-noise)/(stage_reference-noise), then apply the smooth positive part to z. Add scale-invariance and stage-start-equals-one tests."),
        ("2. Separate exploration from stopping", "For pilot/diagnostic campaigns, accept every solved BF stage through the fixed cap. Do not compare scalars whose references are refreshed at each BF."),
        ("3. Preserve the useful controls", "The eight completed control cases need not be rerun. Their BF1-BF7 trajectories and direct-fit reference remain valid."),
        ("4. Run a BF1-BF3 hybrid gate first", "Rerun the eight hybrid clean/noisy matched-seed cases only to BF3. Verify stage normalisation, accepted BF additions, decreasing common closure, and saved diagnostics."),
        ("5. Continue the same cases to BF7", "If the BF3 gate passes, run the eight hybrids to BF7 and compare recovery-gap AUC, yielded/high-plastic error, noise sensitivity and seed spread against equal 7/29/57."),
        ("6. Keep projection out", "Projection remains a later experiment. First establish that the raw objective lifecycle and scaling are correct."),
    ]
    y=.84
    for heading,body in sections:
        fig.text(.07,y,heading,fontsize=13,weight="bold"); y-=.043
        fig.text(.085,y,textwrap.fill(body,116),fontsize=10.5,va="top"); y-=.112
    pdf.savefig(fig); plt.close(fig)


def _write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(asdict(rows[0])))
        writer.writeheader(); writer.writerows(asdict(row) for row in rows)


def _csv(path):
    with path.open(newline="",encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot",type=Path,required=True)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--direct-fit",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
