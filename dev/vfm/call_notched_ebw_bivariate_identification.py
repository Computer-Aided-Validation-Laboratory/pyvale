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

ELASTIC_MODULUS_MPA = 190_000.0
POISSONS_RATIO = 0.28
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
    constitutive_law = _create_constitutive_law(args.stress_backend)

    parameter_map_size = np.asarray(
        experiment_data.specimen_geometry.x.shape,
        dtype=np.uint32,
    )
    parameters = {
        "elastic_modulus": ConstitutiveParameter(ELASTIC_MODULUS_MPA, 150_000.0, 250_000.0, parameter_map_size),
        "poissons_ratio": ConstitutiveParameter(POISSONS_RATIO, 0.2, 0.4, parameter_map_size),
        "yield_strength": ConstitutiveParameter(INITIAL_YIELD_STRENGTH_MPA, *YIELD_BOUNDS_MPA, parameter_map_size),
        "hardening_modulus": ConstitutiveParameter(INITIAL_HARDENING_MODULUS_MPA, *HARDENING_BOUNDS_MPA, parameter_map_size),
    }

    phase_0 = IdentificationPhase(
        spatial_parameterisations={
            "elastic_modulus": [SpatialParameterisationKnown()],
            "poissons_ratio": [SpatialParameterisationKnown()],
            "yield_strength": [SpatialParameterisationHomogeneous()],
            "hardening_modulus": [SpatialParameterisationHomogeneous()],
        },
        metrics=[MetricSBVF(mesh_size=SBVF_MESH_SIZE, vf_scaling_fraction=SBVF_SCALING_FRACTION)],
        objective_function=VectorFirstResultPassthrough(),
        optimiser=OptimiserLeastSquares(max_evaluations=PHASE_0_MAX_EVALUATIONS),
    )

    x = experiment_data.specimen_geometry.x
    y = experiment_data.specimen_geometry.y
    force_metric_phase_1 = SliceWiseForceReconstructionMetric(slice_config=SliceConfig(axis="x", num_slices=FORCE_SLICES))
    equilibrium_gap_metrics_phase_1 = [EquilibriumGapMetric(window_size=window) for window in EGI_WINDOWS]
    egi_window_weights = [window[0] for window in EGI_WINDOWS]
    yield_strength_basis = SpatialParameterisationBasisFunction(
        x=x,
        y=y,
        kernel_type="bivariate",
    )

    phase_1 = IdentificationPhase(
        spatial_parameterisations={
            "elastic_modulus": [SpatialParameterisationKnown()],
            "poissons_ratio": [SpatialParameterisationKnown()],
            "yield_strength": [SpatialParameterisationHomogeneous(), yield_strength_basis],
            "hardening_modulus": [SpatialParameterisationHomogeneous()],
        },
        metrics=[force_metric_phase_1, *equilibrium_gap_metrics_phase_1],
        objective_function=CombinedForceAndEquilibriumGapObjective(
            force_weight=FORCE_WEIGHT,
            egi_window_weights=egi_window_weights,
            baseline=CombinedObjectiveBaseline.prior_phase(0),
        ),
        optimiser=OptimiserPatternSearch(
            initial_mesh_size=INITIAL_MESH_SIZE,
            minimum_mesh_size=args.minimum_mesh_size,
            max_iterations=args.max_iterations,
            max_evaluations=args.max_evaluations,
            parallel_workers=args.parallel_workers,
        ),
        refinement_policy=EquilibriumGapBasisGrowthRefinement(
            target=yield_strength_basis,
            max_basis_functions=MAX_BASIS_FUNCTIONS,
            relative_improvement_threshold=MINIMUM_OBJECTIVE_IMPROVEMENT,
        ),
    )

    identification_config = IdentificationConfig(constitutive_law=constitutive_law, parameters=parameters, phases=[phase_0, phase_1])
    result = run_identification(
        experiment_data,
        identification_config,
        input_source=experiment_data_file,
        progress_callback=ConsoleProgressReporter().report if args.show_progress else None,
    )

    output_dir = args.output_root / experiment_data_file.parent.name / "bivariate_gaussian"
    result_file = result.save_to_yaml(output_dir)
    summary = {
        "input": str(experiment_data_file),
        "result": str(result_file),
        "maximum_basis_functions": MAX_BASIS_FUNCTIONS,
        "minimum_objective_improvement": MINIMUM_OBJECTIVE_IMPROVEMENT,
        "force_weight": FORCE_WEIGHT,
        "egi_windows": EGI_WINDOWS,
        "stress_backend": args.stress_backend,
        "phases": ["homogeneous", "bivariate_gaussian"],
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved identification result bundle to {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--minimum-mesh-size", type=float, default=MINIMUM_MESH_SIZE)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--max-evaluations", type=int, default=PHASE_1_MAX_EVALUATIONS, help="Maximum pattern-search evaluations in phase 1.")
    parser.add_argument("--parallel-workers", type=int, default=PARALLEL_WORKERS)
    parser.add_argument("--stress-backend", choices=("numpy", "cython"), default=STRESS_BACKEND, help="Stress-reconstruction backend; overrides STRESS_BACKEND.")
    parser.add_argument("--no-progress", action="store_false", dest="show_progress", default=SHOW_PROGRESS, help="Disable console progress messages during identification.")
    args = parser.parse_args()
    if not 0.0 < args.minimum_mesh_size <= INITIAL_MESH_SIZE:
        parser.error(f"--minimum-mesh-size must lie in (0, {INITIAL_MESH_SIZE}].")
    if args.max_iterations < 1 or args.max_evaluations < 1 or args.parallel_workers < 1:
        parser.error("Iteration, evaluation, and worker counts must be positive.")
    return args


def _create_constitutive_law(stress_backend: str):
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
    return CompiledLinearHardeningLaw(HardeningLinear())


if __name__ == "__main__":
    main()
