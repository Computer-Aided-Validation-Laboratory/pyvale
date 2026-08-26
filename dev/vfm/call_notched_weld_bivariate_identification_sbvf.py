"""Run two-phase SBVF identification with EGI basis refinement for a notched weld."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyvale.vfm import (
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
    ScalarFirstResultRms,
    SliceConfig,
    SliceWiseForceReconstructionMetric,
    SpatialParameterisationBasisFunction,
    SpatialParameterisationHomogeneous,
    SpatialParameterisationKnown,
    VectorFirstResultPassthrough,
    run_identification,
)
from pyvale.vfm.objectivefuncfreandegi import (
    calculate_force_reconstruction_spatiotemporal_rms,
)


DATASET = Path("/home/robh/1_Projects/pyvale-vfm-test-data/notched-weld-data")
INPUT_PATH = DATASET / "prepared"
OUTPUT_ROOT = DATASET / "identification"

ELASTIC_MODULUS_MPA = 190_000.0
POISSONS_RATIO = 0.28
INITIAL_YIELD_STRENGTH_MPA = 360.0
INITIAL_HARDENING_MODULUS_MPA = 3_700.0

YIELD_BOUNDS_MPA = (200.0, 700.0)
HARDENING_BOUNDS_MPA = (500.0, 10_000.0)

EGI_WINDOWS = ((29, 29), (57, 57))
FORCE_SLICES = 63
SBVF_MESH_SIZE = np.asarray((15, 15), dtype=np.uint32)
SBVF_SCALING_FRACTION = 0.3

INITIAL_MESH_SIZE = 0.1
MINIMUM_MESH_SIZE = 5.0e-4
MAX_ITERATIONS = 200
PHASE_0_MAX_EVALUATIONS = 12
PHASE_1_MAX_EVALUATIONS = 5000
PARALLEL_WORKERS = 12
STRESS_BACKEND = "cython"
SHOW_PROGRESS = True
MAX_BASIS_FUNCTIONS = 6
MINIMUM_EGI_IMPROVEMENT = 0.05


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
        "elastic_modulus": ConstitutiveParameter(
            ELASTIC_MODULUS_MPA, 150_000.0, 250_000.0, parameter_map_size,
        ),
        "poissons_ratio": ConstitutiveParameter(
            POISSONS_RATIO, 0.2, 0.4, parameter_map_size,
        ),
        "yield_strength": ConstitutiveParameter(
            INITIAL_YIELD_STRENGTH_MPA, *YIELD_BOUNDS_MPA, parameter_map_size,
        ),
        "hardening_modulus": ConstitutiveParameter(
            INITIAL_HARDENING_MODULUS_MPA,
            *HARDENING_BOUNDS_MPA,
            parameter_map_size,
        ),
    }

    phase_0 = IdentificationPhase(
        spatial_parameterisations={
            "elastic_modulus": [SpatialParameterisationKnown()],
            "poissons_ratio": [SpatialParameterisationKnown()],
            "yield_strength": [SpatialParameterisationHomogeneous()],
            "hardening_modulus": [SpatialParameterisationHomogeneous()],
        },
        metrics=[_create_sbvf(args)],
        objective_function=VectorFirstResultPassthrough(),
        optimiser=OptimiserLeastSquares(
            max_evaluations=PHASE_0_MAX_EVALUATIONS,
        ),
    )

    x = experiment_data.specimen_geometry.x
    y = experiment_data.specimen_geometry.y
    yield_strength_basis = SpatialParameterisationBasisFunction(
        x=x,
        y=y,
        kernel_type="bivariate",
    )
    egi_metrics = [
        EquilibriumGapMetric(
            window_size=window,
            include_optimisation_diagnostics=False,
        )
        for window in EGI_WINDOWS
    ]
    objective_function, optimiser = _create_phase_1_solver(args)
    phase_1 = IdentificationPhase(
        spatial_parameterisations={
            "elastic_modulus": [SpatialParameterisationKnown()],
            "poissons_ratio": [SpatialParameterisationKnown()],
            "yield_strength": [
                SpatialParameterisationHomogeneous(),
                yield_strength_basis,
            ],
            "hardening_modulus": [SpatialParameterisationHomogeneous()],
        },
        metrics=[_create_sbvf(args), *egi_metrics],
        objective_function=objective_function,
        optimiser=optimiser,
        refinement_policy=EquilibriumGapBasisGrowthRefinement(
            target=yield_strength_basis,
            max_basis_functions=MAX_BASIS_FUNCTIONS,
            relative_improvement_threshold=MINIMUM_EGI_IMPROVEMENT,
            egi_window_weights=[window[0] for window in EGI_WINDOWS],
            baseline_phase_index=0,
        ),
    )

    result = run_identification(
        experiment_data,
        IdentificationConfig(
            constitutive_law=constitutive_law,
            parameters=parameters,
            phases=[phase_0, phase_1],
        ),
        input_source=experiment_data_file,
        progress_callback=(
            ConsoleProgressReporter().report if args.show_progress else None
        ),
    )

    run_name = f"bivariate_gaussian_sbvf_{args.perturbation_type}_{args.optimiser}"
    output_dir = args.output_root / experiment_data_file.parent.name / run_name
    result_file = result.save_to_yaml(output_dir)
    diagnostics = _evaluate_final_diagnostics(
        result.final_stress,
        constitutive_law,
        parameter_map_size,
        experiment_data,
    )
    summary = {
        "input": str(experiment_data_file),
        "result": str(result_file),
        "perturbation_type": args.perturbation_type,
        "perturbation_factor": args.perturbation_factor,
        "optimiser": args.optimiser,
        "maximum_basis_functions": MAX_BASIS_FUNCTIONS,
        "minimum_egi_improvement": MINIMUM_EGI_IMPROVEMENT,
        "egi_windows": EGI_WINDOWS,
        "stress_backend": args.stress_backend,
        "phases": ["homogeneous_sbvf", "bivariate_gaussian_sbvf"],
        "final_diagnostics": diagnostics,
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved identification result bundle to {output_dir}")


def _create_sbvf(args: argparse.Namespace) -> MetricSBVF:
    return MetricSBVF(
        mesh_size=SBVF_MESH_SIZE,
        vf_scaling_fraction=SBVF_SCALING_FRACTION,
        perturbation_type=args.perturbation_type,
        perturbation_factor=args.perturbation_factor,
    )


def _create_phase_1_solver(args: argparse.Namespace):
    if args.optimiser == "least_squares":
        return (
            VectorFirstResultPassthrough(),
            OptimiserLeastSquares(max_evaluations=args.max_evaluations),
        )
    return (
        ScalarFirstResultRms(),
        OptimiserPatternSearch(
            initial_mesh_size=INITIAL_MESH_SIZE,
            minimum_mesh_size=args.minimum_mesh_size,
            max_iterations=args.max_iterations,
            max_evaluations=args.max_evaluations,
            parallel_workers=args.parallel_workers,
        ),
    )


def _evaluate_final_diagnostics(
    stress: np.ndarray,
    constitutive_law,
    parameter_map_size: np.ndarray,
    experiment_data: ExperimentData,
) -> dict[str, object]:
    force_metric = SliceWiseForceReconstructionMetric(
        slice_config=SliceConfig(axis="x", num_slices=FORCE_SLICES),
    )
    force_metric.initialise(experiment_data)
    force_result = force_metric.evaluate(
        stress,
        constitutive_law,
        parameter_map_size,
        {},
        experiment_data,
    )
    egi_values = []
    for window in EGI_WINDOWS:
        metric = EquilibriumGapMetric(window_size=window)
        metric.initialise(experiment_data)
        egi_values.append(
            metric.evaluate_equilibrium_gap(stress).weighted_spatiotemporal_rms
        )
    return {
        "force_reconstruction_rms": (
            calculate_force_reconstruction_spatiotemporal_rms(force_result)
        ),
        "egi_spatiotemporal_rms": egi_values,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--perturbation-type",
        choices=("constitutive_parameter", "dof"),
        default="constitutive_parameter",
    )
    parser.add_argument("--perturbation-factor", type=float, default=0.15)
    parser.add_argument(
        "--optimiser",
        choices=("least_squares", "pattern_search"),
        default="least_squares",
    )
    parser.add_argument("--minimum-mesh-size", type=float, default=MINIMUM_MESH_SIZE)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--max-evaluations", type=int, default=PHASE_1_MAX_EVALUATIONS)
    parser.add_argument("--parallel-workers", type=int, default=PARALLEL_WORKERS)
    parser.add_argument(
        "--stress-backend",
        choices=("numpy", "cython"),
        default=STRESS_BACKEND,
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="show_progress",
        default=SHOW_PROGRESS,
    )
    args = parser.parse_args()
    if not 0.0 < args.perturbation_factor < 1.0:
        parser.error("--perturbation-factor must lie in (0, 1).")
    if not 0.0 < args.minimum_mesh_size <= INITIAL_MESH_SIZE:
        parser.error(
            f"--minimum-mesh-size must lie in (0, {INITIAL_MESH_SIZE}]."
        )
    if args.max_iterations < 1 or args.max_evaluations < 1:
        parser.error("Iteration and evaluation limits must be positive.")
    if args.parallel_workers < 1:
        parser.error("--parallel-workers must be positive.")
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
