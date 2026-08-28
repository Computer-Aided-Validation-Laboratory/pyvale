"""Run four reduced fixed-geometry synthetic notched-EBW VFM tests.

The EGI-29/57/FRE objective is identical to the retained control.  Gaussian
centres, variances and angles are fixed; only homogeneous yield, Gaussian
heights, and optionally homogeneous hardening are active.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pyvale.vfm import (
    CombinedForceAndEquilibriumGapObjective,
    CombinedObjectiveBaseline,
    ConstitutiveParameter,
    DegreeOfFreedom,
    EquilibriumGapMetric,
    ExperimentData,
    OptimiserPatternSearch,
    SliceConfig,
    SliceWiseForceReconstructionMetric,
    SpatialParameterisationBasisFunction,
    SpatialParameterisationHomogeneous,
    SpatialParameterisationKnown,
    load_identification_result,
)
from pyvale.vfm.postprocessing import (
    evaluate_snapshot_parameter_maps,
    load_constitutive_law_from_result,
    load_known_parameter_maps,
)
from pyvale.vfm.spatialparambasisfuncs import BasisFunctionKernelBivariate
from pyvale.vfm.spatialparam import PhaseSpatialState


DATASET = Path("/home/robh/1_Projects/pyvale-vfm-test-data/notched-ebw/synthetic-fe/wdbn1-idealised-yield/pyvale-vfm")
CONTROL = DATASET / "identification/prepared/egi_window_baseline_15500_20260827/identification_result.yaml"
ORACLE_PARAMETERS = Path("dev/vfm/output/notched_ebw_capacity_checks/roi_min1_max5/basis_parameters.csv")
OUTPUT = Path("dev/vfm/output/notched_ebw_fixed_geometry_20260827")
WINDOWS = (29, 57)
FORCE_WEIGHT = 0.1


def main() -> None:
    args = _args()
    experiment = ExperimentData.load_from_file(args.input / "experiment_data.yaml")
    known = load_known_parameter_maps(args.input / "known_parameter_maps.npz")
    if known is None:
        raise RuntimeError("Known synthetic maps are required.")
    control = load_identification_result(args.control)
    phase0 = evaluate_snapshot_parameter_maps(control.history.phases[0].final_snapshot, experiment)
    initial_yield = float(np.nanmean(phase0["yield_strength"]))
    initial_hardening = args.initial_hardening
    baselines = _baselines(control)
    cases = (
        ("A_oracle_geometry_hardening_fixed", _oracle_kernels(), True),
        ("B_oracle_geometry_hardening_free", _oracle_kernels(), False),
        ("C_identified_geometry_hardening_fixed", _identified_kernels(control), True),
        ("D_identified_geometry_hardening_free", _identified_kernels(control), False),
    )
    summary = []
    for name, kernels, fixed_hardening in cases:
        if args.case is not None and args.case != name[0]:
            continue
        output = args.output / name
        result = _run_case(
            output, experiment, known, control, baselines, kernels,
            fixed_hardening=fixed_hardening, initial_yield=initial_yield,
            initial_hardening=initial_hardening, args=args,
        )
        summary.append(result)
        print(f"{name}: cost={result['objective']:.8g}, evaluations={result['evaluations']}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATASET / "prepared")
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--case", choices=("A", "B", "C", "D"))
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument("--max-evaluations", type=int, default=1800)
    parser.add_argument("--parallel-workers", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--initial-hardening", type=float, default=4000.0,
                        help="Controlled start for free-hardening tests.")
    return parser.parse_args()


def _baselines(control) -> tuple[np.ndarray, float]:
    accepted = [solve for solve in control.history.phases[1].solve_results if solve.accepted]
    component = accepted[-1].final_objective["components"]
    return np.asarray(component["egi_baselines"], dtype=float), float(component["force_baseline"])


def _oracle_kernels() -> list[dict[str, float]]:
    with ORACLE_PARAMETERS.open() as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["requested_bases"]) == 5]
    configuration = rows[0]["configuration"]
    return [_csv_kernel(row) for row in rows if row["configuration"] == configuration]


def _csv_kernel(row: dict[str, str]) -> dict[str, float]:
    return {
        "x": float(row["centre_x_mm"]), "y": float(row["centre_y_mm"]),
        "vx": float(row["sigma_major_mm"]) ** 2,
        "vy": float(row["sigma_minor_mm"]) ** 2,
        "angle": float(row["angle_radians"]), "height": float(row["height_mpa"]),
    }


def _identified_kernels(control) -> list[dict[str, float]]:
    kernels = []
    for item in control.history.phases[1].final_snapshot.spatial_parameterisations["yield_strength"]:
        if item.summary.get("kind") == "basis_functions":
            for kernel in item.summary.get("kernels", []):
                kernels.append({
                    "x": float(kernel["centre"][0]), "y": float(kernel["centre"][1]),
                    "vx": float(kernel["variance"][0]), "vy": float(kernel["variance"][1]),
                    "angle": float(kernel["angle"]), "height": float(kernel["height"]),
                })
    return kernels


def _run_case(output, experiment, known, control, baselines, kernels, *, fixed_hardening, initial_yield, initial_hardening, args):
    output.mkdir(parents=True, exist_ok=True)
    size = np.asarray(experiment.specimen_geometry.x.shape, dtype=np.uint32)
    parameters = {
        "elastic_modulus": ConstitutiveParameter(known["elastic_modulus"], 150000.0, 250000.0),
        "poissons_ratio": ConstitutiveParameter(known["poissons_ratio"], 0.2, 0.4),
        "yield_strength": ConstitutiveParameter(initial_yield, 200.0, 2000.0, size),
        "hardening_modulus": ConstitutiveParameter(4000.0 if fixed_hardening else initial_hardening, 500.0, 10000.0, size),
    }
    basis = _basis(experiment, kernels)
    parameterisations = {
        "elastic_modulus": [SpatialParameterisationKnown()],
        "poissons_ratio": [SpatialParameterisationKnown()],
        "yield_strength": [SpatialParameterisationHomogeneous(), basis],
        "hardening_modulus": [SpatialParameterisationKnown() if fixed_hardening else SpatialParameterisationHomogeneous()],
    }
    for name, items in parameterisations.items():
        for item in items:
            item.initialise_from_constitutive_parameter(parameters[name])
    metrics = _metrics(experiment)
    objective = CombinedForceAndEquilibriumGapObjective(
        force_weight=FORCE_WEIGHT, egi_window_weights=WINDOWS,
        egi_baseline_values=baselines[0], force_baseline_value=baselines[1],
    )
    optimiser = OptimiserPatternSearch(
        initial_mesh_size=0.1, minimum_mesh_size=5e-4,
        max_iterations=args.max_iterations, max_evaluations=args.max_evaluations,
        parallel_workers=args.parallel_workers, random_seed=args.random_seed,
    )
    outcome = optimiser.optimise(
        load_constitutive_law_from_result(control), size, parameterisations,
        metrics, objective, experiment,
    )
    state = PhaseSpatialState(outcome.spatial_parameterisations)
    maps = state.evaluate_parameter_maps(size)
    stress = load_constitutive_law_from_result(control).calculate_stress(experiment.strain, maps)
    np.savez_compressed(output / "result.npz", **maps, stress=stress)
    error = maps["yield_strength"] - known["yield_strength"]
    roi = experiment.specimen_geometry.region_of_interest.sample_specimen_mask(experiment.specimen_geometry.x, experiment.specimen_geometry.y)
    result = {
        "name": output.name, "objective": float(outcome.solve_result.final_objective["cost"]),
        "evaluations": int(outcome.solve_result.num_evaluations), "status": outcome.solve_result.status,
        "hardening_fixed": fixed_hardening, "hardening_mpa": float(np.nanmean(maps["hardening_modulus"])),
        "yield_roi_rmse_mpa": float(np.sqrt(np.nanmean(error[roi] ** 2))),
        "components": outcome.solve_result.final_objective.get("components", {}),
        "initial_dofs": outcome.solve_result.initial_dofs, "final_dofs": outcome.solve_result.final_dofs,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, default=_json_default))
    return result


def _basis(experiment, kernels) -> SpatialParameterisationBasisFunction:
    objects = [BasisFunctionKernelBivariate(item["x"], item["y"], item["vx"], item["vy"], item["angle"]) for item in kernels]
    heights = [DegreeOfFreedom(item["height"], -1800.0, 1800.0) for item in kernels]
    return SpatialParameterisationBasisFunction(
        x=experiment.specimen_geometry.x, y=experiment.specimen_geometry.y,
        kernels=objects, heights=heights, kernel_type="bivariate",
    )


def _metrics(experiment):
    egi = [EquilibriumGapMetric(window_size=(size, size)) for size in WINDOWS]
    for metric in egi:
        metric.initialise(experiment)
    force = SliceWiseForceReconstructionMetric(slice_config=SliceConfig(axis="x", num_slices=63))
    force.initialise(experiment)
    return [force, *egi]


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


if __name__ == "__main__":
    main()
