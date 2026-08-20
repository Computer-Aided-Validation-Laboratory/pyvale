"""Run auditable slice-wise VFM verification against prepared FE data.

Unlike ``call_vfm_sw_refine_clean.py``, this caller accepts explicit physical
slice boundaries and can hold the hardening modulus known.  It is intended for
noise-free FE verification cases where material interfaces must coincide with
slice support-cell edges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyvale.vfm import (
    ConsoleProgressReporter,
    ConstitutiveParameter,
    ExperimentData,
    HardeningLinear,
    IdentificationConfig,
    IdentificationPhase,
    IsotropicVonMisesElastoplasticity,
    SliceConfig,
    SliceMergeSplitRefinement,
    SliceWiseForceReconstructionMetric,
    SliceWiseIndependentLeastSquares,
    SliceWiseSpatialParameterisation,
    SpatialParameterisationKnown,
    SupportSlice,
    VectorWeightedObjective,
    run_identification,
)


DEFAULT_INPUT = Path(
    "/home/robh/1_Projects/pyvale-vfm-test-data/notched-weld-vertical-slice/prepared"
)
DEFAULT_BOUNDARIES = (12.5, 35.0, 40.0, 62.5)


def main() -> None:
    args = _parse_args()
    experiment_file = args.input / "experiment_data.yaml" if args.input.is_dir() else args.input
    experiment_data = ExperimentData.load_from_file(experiment_file)
    parameter_shape = np.asarray(experiment_data.specimen_geometry.x.shape, dtype=np.uint32)
    known_maps = _load_known_maps(experiment_file.parent, parameter_shape)
    slice_config = SliceConfig(
        axis=args.axis,
        boundaries=(np.asarray(args.boundaries, dtype=np.float64) if args.boundaries else None),
        num_slices=(None if args.boundaries else args.num_slices),
    )
    support = SupportSlice(slice_config=slice_config)
    parameters = {
        "elastic_modulus": _parameter(known_maps["elastic_modulus"], 150_000.0, 250_000.0, parameter_shape),
        "poissons_ratio": _parameter(known_maps["poissons_ratio"], 0.2, 0.4, parameter_shape),
        "yield_strength": _parameter(np.full(tuple(parameter_shape), args.yield_initial), args.yield_lower, args.yield_upper, parameter_shape),
        "hardening_modulus": _parameter(
            known_maps["hardening_modulus"] if args.hardening_mode == "known" else np.full(tuple(parameter_shape), args.hardening_initial),
            args.hardening_lower,
            args.hardening_upper,
            parameter_shape,
        ),
    }
    spatial = {
        "elastic_modulus": [SpatialParameterisationKnown()],
        "poissons_ratio": [SpatialParameterisationKnown()],
        "yield_strength": [SliceWiseSpatialParameterisation(support=support)],
        "hardening_modulus": [
            SpatialParameterisationKnown()
            if args.hardening_mode == "known"
            else SliceWiseSpatialParameterisation(support=support)
        ],
    }
    refinement = None
    if args.max_refinements:
        refinement = SliceMergeSplitRefinement(
            target=support,
            merge_parameter_tolerance=args.merge_parameter_tolerance,
            split_error_threshold=args.split_force_error_threshold,
            max_refinements=args.max_refinements,
        )
    phase = IdentificationPhase(
        spatial_parameterisations=spatial,
        metrics=[SliceWiseForceReconstructionMetric(support=support)],
        objective_function=VectorWeightedObjective(),
        optimiser=SliceWiseIndependentLeastSquares(),
        refinement_policy=refinement,
    )
    result = run_identification(
        experiment_data,
        IdentificationConfig(
            constitutive_law=IsotropicVonMisesElastoplasticity(HardeningLinear()),
            parameters=parameters,
            phases=[phase],
        ),
        input_source=experiment_file,
        progress_callback=ConsoleProgressReporter().report if args.show_progress else None,
    )
    output_dir = args.output_root / args.name
    result_file = result.save_to_yaml(output_dir)
    summary = _build_summary(result, known_maps, args, result_file)
    (output_dir / "verification_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _parameter(value: np.ndarray, lower: float, upper: float, shape: np.ndarray) -> ConstitutiveParameter:
    return ConstitutiveParameter(value, lower, upper, shape)


def _load_known_maps(prepared: Path, shape: np.ndarray) -> dict[str, np.ndarray]:
    path = prepared / "known_parameter_maps.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Verification caller requires known maps: {path}")
    with np.load(path) as loaded:
        maps = {name: np.asarray(loaded[name], dtype=np.float64) for name in ("elastic_modulus", "poissons_ratio", "hardening_modulus")}
    expected_shape = tuple(shape)
    for name, value in maps.items():
        if value.shape != expected_shape:
            raise ValueError(f"Known map '{name}' shape {value.shape} does not match data shape {expected_shape}.")
    return maps


def _build_summary(result, known_maps: dict[str, np.ndarray], args: argparse.Namespace, result_file: Path) -> dict:
    identified = result.parameter_maps
    return {
        "result": str(result_file),
        "axis": args.axis,
        "hardening_mode": args.hardening_mode,
        "requested_boundaries_mm": list(args.boundaries) if args.boundaries else None,
        "num_slices": args.num_slices if not args.boundaries else len(args.boundaries) - 1,
        "yield_strength_mpa": _finite_range(identified["yield_strength"]),
        "hardening_modulus_mpa": _finite_range(identified["hardening_modulus"]),
        "known_elastic_modulus_mpa": _finite_range(known_maps["elastic_modulus"]),
        "known_poissons_ratio": _finite_range(known_maps["poissons_ratio"]),
    }


def _finite_range(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    return {"min": float(np.min(finite)), "max": float(np.max(finite)), "mean": float(np.mean(finite))}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_INPUT.parent / "identification")
    parser.add_argument("--name", default="simultaneous-yield-hardening")
    parser.add_argument("--axis", choices=("x", "y"), default="x")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--boundaries", nargs="+", type=float)
    selector.add_argument("--num-slices", type=int)
    parser.add_argument("--hardening-mode", choices=("known", "slicewise"), default="slicewise")
    parser.add_argument("--yield-initial", type=float, default=390.0)
    parser.add_argument("--yield-lower", type=float, default=300.0)
    parser.add_argument("--yield-upper", type=float, default=500.0)
    parser.add_argument("--hardening-initial", type=float, default=3700.0)
    parser.add_argument("--hardening-lower", type=float, default=1000.0)
    parser.add_argument("--hardening-upper", type=float, default=8000.0)
    parser.add_argument("--max-refinements", type=int, default=0)
    parser.add_argument("--merge-parameter-tolerance", type=float, default=0.05)
    parser.add_argument("--split-force-error-threshold", type=float, default=0.1)
    parser.add_argument("--no-progress", action="store_false", dest="show_progress", default=True)
    args = parser.parse_args()
    if args.boundaries is None and args.num_slices is None:
        args.boundaries = DEFAULT_BOUNDARIES
    if args.boundaries is not None and len(args.boundaries) < 2:
        parser.error("--boundaries requires at least two coordinates.")
    if args.boundaries is None and (args.num_slices is None or args.num_slices < 1):
        parser.error("provide --boundaries or a positive --num-slices.")
    return args


if __name__ == "__main__":
    main()
