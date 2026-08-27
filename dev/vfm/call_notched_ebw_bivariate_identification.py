"""Run two-phase bivariate-Gaussian VFM identification for a notched weld.

The first phase identifies homogeneous yield strength and hardening modulus.
The second phase grows bivariate Gaussian basis functions in the yield-strength
field from the combined equilibrium-gap indicator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyvale.vfm import (
    CombinedForceAndEquilibriumGapObjective,
    CombinedObjectiveBaseline,
    ConsoleProgressReporter,
    ConstitutiveParameter,
    EquilibriumGapBasisGrowthRefinement,
    EquilibriumGapMetric,
    ExperimentData,
    HardeningLinear,
    IdentificationConfig,
    IdentificationPhase,
    IsotropicVonMisesElastoplasticity,
    MetricSBVF,
    OptimiserLeastSquares,
    OptimiserPatternSearch,
    SliceConfig,
    SliceWiseForceReconstructionMetric,
    SpatialParameterisationBasisFunction,
    SpatialParameterisationHomogeneous,
    SpatialParameterisationKnown,
    VectorFirstResultPassthrough,
    run_identification,
)


# =============================================================================
# User inputs
# =============================================================================

DATASET = Path("/home/robh/1_Projects/pyvale-vfm-test-data/notched-ebw/synthetic-fe/wdbn1-idealised-yield/pyvale-vfm")
INPUT_PATH = DATASET / "prepared"
OUTPUT_ROOT = DATASET / "identification"

ELASTIC_MODULUS_MPA = 210_000.0
POISSONS_RATIO = 0.3
INITIAL_YIELD_STRENGTH_MPA = 360.0
INITIAL_HARDENING_MODULUS_MPA = 3_700.0

YIELD_BOUNDS_MPA = (200.0, 2000.0)
HARDENING_BOUNDS_MPA = (500.0, 10_000.0)

FORCE_WEIGHT = 0.1
EGI_WINDOWS = ((29, 29), (57, 57))
FORCE_SLICES = 63
SBVF_MESH_SIZE = np.asarray((15, 15), dtype=np.uint32)
SBVF_SCALING_FRACTION = 0.3

INITIAL_MESH_SIZE = 0.1
MINIMUM_MESH_SIZE = 5.0e-4
MAX_ITERATIONS = 200
PHASE_0_MAX_EVALUATIONS = 12
PHASE_1_MAX_EVALUATIONS = 5000  # eval_count ~ 1 + (iter_count * ((2*dof_count) + 1)) i.e. for 8 dof, 1 + (iter_count * 17)
PARALLEL_WORKERS = 12
STRESS_BACKEND = "cython"
SHOW_PROGRESS = True
MAX_BASIS_FUNCTIONS = 6
MINIMUM_OBJECTIVE_IMPROVEMENT = 0.05


def main() -> None:
    args = _parse_args()

    experiment_data_file = (
        args.input / "experiment_data.yaml"
        if args.input.is_dir()
        else args.input
    )
    experiment_data = ExperimentData.load_from_file(experiment_data_file)
    constitutive_law = _create_constitutive_law(
        args.stress_backend,
        minimum_yield_strength=YIELD_BOUNDS_MPA[0],
    )

    parameter_map_size = np.asarray(
        experiment_data.specimen_geometry.x.shape,
        dtype=np.uint32,
    )
    parameters = {
        "elastic_modulus": ConstitutiveParameter(args.elastic_modulus, 150_000.0, 250_000.0, parameter_map_size),
        "poissons_ratio": ConstitutiveParameter(args.poissons_ratio, 0.2, 0.4, parameter_map_size),
        "yield_strength": ConstitutiveParameter(args.initial_yield_strength, *YIELD_BOUNDS_MPA, parameter_map_size),
        "hardening_modulus": ConstitutiveParameter(args.initial_hardening_modulus, *HARDENING_BOUNDS_MPA, parameter_map_size),
    }

    phase_0 = IdentificationPhase(
        spatial_parameterisations={
            "elastic_modulus": [SpatialParameterisationKnown()],
            "poissons_ratio": [SpatialParameterisationKnown()],
            "yield_strength": [SpatialParameterisationHomogeneous()],
            "hardening_modulus": [
                SpatialParameterisationKnown()
                if args.fix_hardening
                else SpatialParameterisationHomogeneous()
            ],
        },
        metrics=[MetricSBVF(mesh_size=SBVF_MESH_SIZE, vf_scaling_fraction=SBVF_SCALING_FRACTION)],
        objective_function=VectorFirstResultPassthrough(),
        optimiser=OptimiserLeastSquares(max_evaluations=args.phase_0_max_evaluations),
    )

    x = experiment_data.specimen_geometry.x
    y = experiment_data.specimen_geometry.y
    force_metric_phase_1 = SliceWiseForceReconstructionMetric(slice_config=SliceConfig(axis="x", num_slices=args.force_slices))
    equilibrium_gap_metrics_phase_1 = [EquilibriumGapMetric(window_size=window) for window in args.egi_windows]
    egi_window_weights = [window[0] for window in args.egi_windows]
    yield_strength_basis = SpatialParameterisationBasisFunction(
        x=x,
        y=y,
        kernel_type="bivariate",
        centre_bounds_span_factor=args.centre_bounds_span_factor,
    )

    phase_1 = IdentificationPhase(
        spatial_parameterisations={
            "elastic_modulus": [SpatialParameterisationKnown()],
            "poissons_ratio": [SpatialParameterisationKnown()],
            "yield_strength": [SpatialParameterisationHomogeneous(), yield_strength_basis],
            "hardening_modulus": [
                SpatialParameterisationKnown()
                if args.fix_hardening
                else SpatialParameterisationHomogeneous()
            ],
        },
        metrics=[force_metric_phase_1, *equilibrium_gap_metrics_phase_1],
        objective_function=CombinedForceAndEquilibriumGapObjective(
            force_weight=args.force_weight,
            egi_window_weights=egi_window_weights,
            baseline=CombinedObjectiveBaseline.prior_phase(0),
        ),
        optimiser=OptimiserPatternSearch(
            initial_mesh_size=args.initial_mesh_size,
            minimum_mesh_size=args.minimum_mesh_size,
            max_iterations=args.max_iterations,
            max_evaluations=args.max_evaluations,
            parallel_workers=args.parallel_workers,
        ),
        refinement_policy=EquilibriumGapBasisGrowthRefinement(
            target=yield_strength_basis,
            max_basis_functions=args.max_basis_functions,
            relative_improvement_threshold=args.minimum_objective_improvement,
        ),
    )

    identification_config = IdentificationConfig(constitutive_law=constitutive_law, parameters=parameters, phases=[phase_0, phase_1])
    result = run_identification(
        experiment_data,
        identification_config,
        input_source=experiment_data_file,
        progress_callback=ConsoleProgressReporter().report if args.show_progress else None,
    )

    output_dir = args.output_root / args.run_name
    result_file = result.save_to_yaml(output_dir)
    summary = {
        "input": str(experiment_data_file),
        "result": str(result_file),
        "run_name": args.run_name,
        "elastic_modulus_mpa": args.elastic_modulus,
        "poissons_ratio": args.poissons_ratio,
        "initial_yield_strength_mpa": args.initial_yield_strength,
        "initial_hardening_modulus_mpa": args.initial_hardening_modulus,
        "phase_0_max_evaluations": args.phase_0_max_evaluations,
        "maximum_basis_functions": args.max_basis_functions,
        "minimum_objective_improvement": args.minimum_objective_improvement,
        "initial_mesh_size": args.initial_mesh_size,
        "centre_bounds_span_factor": args.centre_bounds_span_factor,
        "hardening_fixed": args.fix_hardening,
        "force_weight": args.force_weight,
        "force_slices": args.force_slices,
        "egi_windows": args.egi_windows,
        "stress_backend": args.stress_backend,
        "phases": ["homogeneous", "bivariate_gaussian"],
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved identification result bundle to {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-name", default="prepared/bivariate_gaussian")
    parser.add_argument("--elastic-modulus", type=float, default=ELASTIC_MODULUS_MPA)
    parser.add_argument("--poissons-ratio", type=float, default=POISSONS_RATIO)
    parser.add_argument("--initial-yield-strength", type=float, default=INITIAL_YIELD_STRENGTH_MPA)
    parser.add_argument("--initial-hardening-modulus", type=float, default=INITIAL_HARDENING_MODULUS_MPA)
    parser.add_argument("--phase-0-max-evaluations", type=int, default=PHASE_0_MAX_EVALUATIONS)
    parser.add_argument("--force-weight", type=float, default=FORCE_WEIGHT)
    parser.add_argument("--force-slices", type=int, default=FORCE_SLICES)
    parser.add_argument("--egi-windows", type=_parse_egi_windows, default=EGI_WINDOWS, help="Comma-separated odd square window sizes, e.g. 15,29,41.")
    parser.add_argument("--max-basis-functions", type=int, default=MAX_BASIS_FUNCTIONS)
    parser.add_argument("--minimum-objective-improvement", type=float, default=MINIMUM_OBJECTIVE_IMPROVEMENT)
    parser.add_argument("--initial-mesh-size", type=float, default=INITIAL_MESH_SIZE, help="Initial normalised pattern-search mesh size.")
    parser.add_argument("--centre-bounds-span-factor", type=float, default=1.0, help="Multiplier applied to the coordinate span used for Gaussian centre bounds.")
    parser.add_argument("--fix-hardening", action="store_true", help="Hold the supplied initial hardening modulus fixed in both phases.")
    parser.add_argument("--minimum-mesh-size", type=float, default=MINIMUM_MESH_SIZE)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--max-evaluations", type=int, default=PHASE_1_MAX_EVALUATIONS, help="Maximum pattern-search evaluations in phase 1.")
    parser.add_argument("--parallel-workers", type=int, default=PARALLEL_WORKERS)
    parser.add_argument("--stress-backend", choices=("numpy", "cython"), default=STRESS_BACKEND, help="Stress-reconstruction backend; overrides STRESS_BACKEND.")
    parser.add_argument("--no-progress", action="store_false", dest="show_progress", default=SHOW_PROGRESS, help="Disable console progress messages during identification.")
    args = parser.parse_args()
    if not 0.0 < args.initial_mesh_size <= 1.0:
        parser.error("--initial-mesh-size must lie in (0, 1].")
    if not 0.0 < args.minimum_mesh_size <= args.initial_mesh_size:
        parser.error("--minimum-mesh-size must lie in (0, initial mesh size].")
    if args.max_iterations < 1 or args.max_evaluations < 1 or args.parallel_workers < 1 or args.phase_0_max_evaluations < 1:
        parser.error("Iteration, evaluation, and worker counts must be positive.")
    if not 0.0 <= args.force_weight <= 1.0:
        parser.error("--force-weight must lie in [0, 1].")
    if args.force_slices < 2 or args.max_basis_functions < 1:
        parser.error("Force slices must be at least two and max bases must be positive.")
    if not 0.0 <= args.minimum_objective_improvement < 1.0:
        parser.error("--minimum-objective-improvement must lie in [0, 1).")
    if not np.isfinite(args.centre_bounds_span_factor) or args.centre_bounds_span_factor < 1.0:
        parser.error("--centre-bounds-span-factor must be finite and at least 1.")
    return args


def _parse_egi_windows(value: str) -> tuple[tuple[int, int], ...]:
    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not sizes or any(size < 3 or size % 2 == 0 for size in sizes):
        raise argparse.ArgumentTypeError("EGI windows must be odd integers of at least 3.")
    return tuple((size, size) for size in sizes)


def _create_constitutive_law(
    stress_backend: str,
    *,
    minimum_yield_strength: float,
):
    if stress_backend == "numpy":
        return IsotropicVonMisesElastoplasticity(HardeningLinear())
    try:
        from cython_stress_recon.pyvale_adapter import CompiledLinearHardeningLaw
    except ImportError as exc:
        raise RuntimeError(
            "The Cython stress backend was requested but cython-stress-recon "
            "is not installed. Install the sibling project into this Python "
            "environment or select --stress-backend numpy."
        ) from exc
    return CompiledLinearHardeningLaw(
        HardeningLinear(),
        minimum_yield_strength=minimum_yield_strength,
    )


if __name__ == "__main__":
    main()
