"""Run two-phase Gaussian VFM identification for a notched weld.

The first phase identifies homogeneous yield strength and hardening modulus.
The second phase grows Gaussian basis functions in the yield-strength field,
with selectable conventional/SPD geometry and EGI/sensitivity growth rules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from pyvale.vfm import (
    CombinedForceAndEquilibriumGapObjective,
    CombinedObjectiveBaseline,
    ConsoleProgressReporter,
    ConstitutiveParameter,
    EquilibriumGapBasisGrowthRefinement,
    EquilibriumGapMetric,
    AutomaticEgiSupportPreparation,
    EgiSupportBankConfig,
    EgiSupportInformationSelectionConfig,
    EgiSignalSelectionConfig,
    ExperimentData,
    FixedEgiSupportPreparation,
    HardeningLinear,
    IdentificationConfig,
    IdentificationPhase,
    IsotropicVonMisesElastoplasticity,
    MetricSBVF,
    OptimiserLeastSquares,
    OptimiserPatternSearch,
    SliceConfig,
    SliceWiseForceReconstructionMetric,
    SensitivityInformationObjective,
    SensitivityInformationObjectiveConfig,
    SensitivityGatedEgiObjective,
    SensitivityGatedObjectiveConfig,
    SensitivitySpatialWeightingConfig,
    SensitivityCorrectionBasisGrowthRefinement,
    SimpleEgiSupportPreparation,
    SpatialParameterisationBasisFunction,
    SpatialParameterisationHomogeneous,
    SpatialParameterisationKnown,
    VectorFirstResultPassthrough,
    run_identification,
)
from pyvale.vfm.egisupports import resolve_physical_egi_supports
from pyvale.vfm.loadregimes import resolve_load_regimes
from pyvale.vfm.residualblocks import ResidualBlockSpec
from diagnostic_artifacts import DiagnosticArtifactWriter
from pyvale.vfm.objectivefuncmaterialinformation import (
    MaterialFeatureReduction,
    MaterialFeatureReference,
    MaterialFeatureTerm,
    MaterialInformationObjective,
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
# Stop once complete polls cannot improve the solution at roughly 0.1% of
# each DOF's configured normalisation span. This criterion is independent of
# objective scaling; callers can still tighten it with --minimum-mesh-size.
MINIMUM_MESH_SIZE = 1.0e-3
OBJECTIVE_RELATIVE_TOLERANCE = 1.0e-4
MAX_ITERATIONS = 200
PHASE_0_MAX_EVALUATIONS = 12
# Six bivariate Gaussian bases give 38 phase-1 DOFs.  200 complete pattern
# search polls can therefore require 1 + 200 * (2 * 38 + 1) = 15,401
# evaluations; keep the evaluation ceiling above this so it does not truncate
# the configured iteration budget during configuration investigations.
PHASE_1_MAX_EVALUATIONS = 15_500
PARALLEL_WORKERS = 12
RANDOM_SEED = 0
STRESS_BACKEND = "cython"
SHOW_PROGRESS = True
MAX_BASIS_FUNCTIONS = 6
MINIMUM_OBJECTIVE_IMPROVEMENT = 0.05
SENSITIVITY_PERTURBATION_FACTOR = 0.15
SENSITIVITY_WEIGHT_FLOOR = 0.1
MULTISTART_OFFSET_FRACTION = 0.10
MULTISTART_SCREENING_ITERATIONS = 10


def main() -> None:
    args = _parse_args()
    output_dir = args.output_root / args.run_name

    experiment_data_file = (
        args.input / "experiment_data.yaml"
        if args.input.is_dir()
        else args.input
    )
    experiment_data = ExperimentData.load_from_file(experiment_data_file)
    noise_diagnostics = _apply_artificial_noise(experiment_data, args)
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
    force_metric_phase_1 = SliceWiseForceReconstructionMetric(
        slice_config=SliceConfig(axis=args.force_axis, num_slices=args.force_slices)
    )
    fft_groups = [None] * len(args.egi_windows)
    if args.egi_fft_groups == "split-broad" and len(fft_groups) > 1:
        fft_groups = ["local"] * (len(fft_groups) - 1) + ["broad"]
    elif args.egi_fft_groups == "separate":
        fft_groups = [f"support-{index}" for index in range(len(fft_groups))]
    equilibrium_gap_metrics_phase_1 = [
        EquilibriumGapMetric(
            window_size=window,
            fft_dtype=args.egi_fft_dtype,
            fft_batch_group=group,
            # The gated objective consumes residual maps directly. Avoid
            # the temporal RMS map during candidates; accepted/reference
            # paths can reconstruct it from the residual. The global RMS
            # scalar remains required to resolve the prior-phase baseline.
            compute_temporal_rms=not args.egi_skip_derived_diagnostics,
            compute_spatiotemporal_rms=True,
        )
        for window, group in zip(args.egi_windows, fft_groups, strict=True)
    ]
    egi_window_weights = (
        [window[0] for window in args.egi_windows]
        if args.egi_window_weights is None
        else args.egi_window_weights
    )
    yield_strength_basis = SpatialParameterisationBasisFunction(
        x=x,
        y=y,
        kernel_type=args.kernel_type,
        centre_bounds_span_factor=args.centre_bounds_span_factor,
    )

    refinement_policy_type = (
        EquilibriumGapBasisGrowthRefinement
        if args.basis_growth_policy == "egi_peak"
        else SensitivityCorrectionBasisGrowthRefinement
    )
    refinement_options = {
        "target": yield_strength_basis,
        "max_basis_functions": args.max_basis_functions,
        "relative_improvement_threshold": args.minimum_objective_improvement,
        "smoothing_points": args.refinement_smoothing_points,
        "multistart_enabled": args.multistart_basis_placement,
        "multistart_offset_fraction": args.multistart_offset_fraction,
        "multistart_screening_iterations": args.multistart_screening_iterations,
        "fixed_basis_trajectory": args.fixed_basis_trajectory,
    }
    if args.objective_config is not None:
        # The refinement policy consumes EGI directly, independently of the
        # hybrid scalar wrapper, so give it the same prior-phase scaling.
        refinement_options.update({
            "baseline_phase_index": 0,
            "egi_window_weights": egi_window_weights,
        })
    if args.basis_growth_policy == "sensitivity_correction":
        refinement_options.update(
            {
                "sensitivity_perturbation_factor": (
                    args.correction_sensitivity_perturbation_factor
                ),
                "correction_feature_fraction": args.correction_feature_fraction,
            }
        )

    if (
        args.data_driven_objective_config is not None
        or args.simple_data_driven_objective_config is not None
    ):
        # Automatic preparation installs exactly fine/middle/broad metrics.
        egi_window_weights = [1.0, 1.0, 1.0]
        refinement_options["egi_window_weights"] = egi_window_weights
        refinement_options["baseline_phase_index"] = 0

    global_objective = CombinedForceAndEquilibriumGapObjective(
        force_weight=args.force_weight,
        egi_window_weights=egi_window_weights,
        baseline=CombinedObjectiveBaseline.prior_phase(0),
        spatial_weighting=(
            SensitivitySpatialWeightingConfig(
                perturbation_factor=args.sensitivity_perturbation_factor,
                weight_floor=args.sensitivity_weight_floor,
            )
            if args.sensitivity_spatial_weighting
            else None
        ),
    )
    phase_preparation = None
    if (
        args.data_driven_objective_config is None
        and args.simple_data_driven_objective_config is None
    ):
        phase_1_objective = _create_phase_1_objective(
            args.objective_config, global_objective
        )
    elif args.data_driven_objective_config is not None:
        payload = json.loads(
            args.data_driven_objective_config.expanduser().resolve().read_text(
                encoding="utf-8"
            )
        )
        phase_1_objective, phase_preparation = (
            _create_data_driven_phase_1_objective(
                payload,
                timestep_count=experiment_data.strain.shape[0],
                basis_growth_objective=global_objective,
                diagnostic_callback=DiagnosticArtifactWriter(
                    output_dir / "diagnostic_artifacts"
                ),
            )
        )
    else:
        payload = json.loads(
            args.simple_data_driven_objective_config.expanduser().resolve().read_text(
                encoding="utf-8"
            )
        )
        phase_1_objective, phase_preparation = (
            _create_simple_data_driven_phase_1_objective(
                payload,
                x=x,
                y=y,
                basis_growth_objective=global_objective,
                diagnostic_callback=DiagnosticArtifactWriter(
                    output_dir / "diagnostic_artifacts"
                ),
            )
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
        objective_function=phase_1_objective,
        optimiser=OptimiserPatternSearch(
            initial_mesh_size=args.initial_mesh_size,
            minimum_mesh_size=args.minimum_mesh_size,
            max_iterations=args.max_iterations,
            max_evaluations=args.max_evaluations,
            objective_relative_tolerance=args.objective_relative_tolerance,
            parallel_workers=args.parallel_workers,
            random_seed=args.random_seed,
        ),
        refinement_policy=refinement_policy_type(**refinement_options),
        phase_preparation=phase_preparation,
    )

    identification_config = IdentificationConfig(constitutive_law=constitutive_law, parameters=parameters, phases=[phase_0, phase_1])
    result = run_identification(
        experiment_data,
        identification_config,
        input_source=experiment_data_file,
        progress_callback=ConsoleProgressReporter().report if args.show_progress else None,
    )

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
        "fixed_basis_trajectory": args.fixed_basis_trajectory,
        "initial_mesh_size": args.initial_mesh_size,
        "minimum_mesh_size": args.minimum_mesh_size,
        "objective_relative_tolerance": args.objective_relative_tolerance,
        "centre_bounds_span_factor": args.centre_bounds_span_factor,
        "kernel_type": args.kernel_type,
        "basis_growth_policy": args.basis_growth_policy,
        "correction_sensitivity_perturbation_factor": (
            args.correction_sensitivity_perturbation_factor
        ),
        "correction_feature_fraction": args.correction_feature_fraction,
        "hardening_fixed": args.fix_hardening,
        "sensitivity_spatial_weighting": args.sensitivity_spatial_weighting,
        "sensitivity_perturbation_factor": args.sensitivity_perturbation_factor,
        "sensitivity_weight_floor": args.sensitivity_weight_floor,
        "multistart_basis_placement": args.multistart_basis_placement,
        "multistart_offset_fraction": args.multistart_offset_fraction,
        "multistart_screening_iterations": args.multistart_screening_iterations,
        "force_weight": args.force_weight,
        "objective_config": (
            None if args.objective_config is None else str(args.objective_config)
        ),
        "data_driven_objective_config": (
            None
            if args.data_driven_objective_config is None
            else str(args.data_driven_objective_config)
        ),
        "simple_data_driven_objective_config": (
            None
            if args.simple_data_driven_objective_config is None
            else str(args.simple_data_driven_objective_config)
        ),
        "force_slices": args.force_slices,
        "force_axis": args.force_axis,
        "egi_windows": args.egi_windows,
        "egi_window_weights": egi_window_weights,
        "egi_fft_dtype": args.egi_fft_dtype,
        "egi_fft_groups": args.egi_fft_groups,
        "egi_skip_derived_diagnostics": args.egi_skip_derived_diagnostics,
        "artificial_noise": noise_diagnostics,
        "stress_backend": args.stress_backend,
        "random_seed": args.random_seed,
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
    parser.add_argument(
        "--objective-config",
        type=Path,
        default=None,
        help=(
            "Optional frozen hybrid-objective JSON from the offline screen. "
            "Omit it to use the existing combined EGI/FRE objective."
        ),
    )
    parser.add_argument(
        "--data-driven-objective-config",
        type=Path,
        default=None,
        help=(
            "JSON configuration for automatic EGI support selection and the "
            "frozen sensitivity-information phase-1 objective."
        ),
    )
    parser.add_argument(
        "--simple-data-driven-objective-config",
        type=Path,
        default=None,
        help=(
            "JSON configuration for the direct-SNR EGI selector and frozen "
            "two-perturbation sensitivity-gated objective."
        ),
    )
    parser.add_argument("--force-slices", type=int, default=FORCE_SLICES)
    parser.add_argument(
        "--force-axis",
        choices=("x", "y"),
        default="x",
        help="Longitudinal/loading axis for the slice-wise force-reconstruction metric.",
    )
    parser.add_argument("--egi-windows", type=_parse_egi_windows, default=EGI_WINDOWS, help="Comma-separated odd square window sizes, e.g. 15,29,41.")
    parser.add_argument(
        "--egi-fft-dtype",
        choices=("float64", "float32"),
        default="float32",
        help="FFT arithmetic used by EGI convolution; use float64 to restore the historical path.",
    )
    parser.add_argument(
        "--egi-fft-groups",
        choices=("all", "split-broad", "separate"),
        default="split-broad",
        help="Share stress FFTs across all supports, split the largest support, or evaluate each separately.",
    )
    parser.add_argument(
        "--egi-skip-derived-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip the derived temporal EGI RMS map while retaining objective residual maps and the required global RMS scalar.",
    )
    parser.add_argument(
        "--egi-window-weights",
        type=_parse_float_tuple,
        default=None,
        help="Optional comma-separated objective weights matching --egi-windows.",
    )
    parser.add_argument("--max-basis-functions", type=int, default=MAX_BASIS_FUNCTIONS)
    parser.add_argument(
        "--kernel-type",
        choices=("bivariate", "bivariate_spd"),
        default="bivariate",
        help="Gaussian geometry coordinates used in phase 1.",
    )
    parser.add_argument(
        "--basis-growth-policy",
        choices=("egi_peak", "sensitivity_correction"),
        default="egi_peak",
        help="Rule used to place each Gaussian added after a solve.",
    )
    parser.add_argument(
        "--correction-sensitivity-perturbation-factor",
        type=float,
        default=0.01,
        help="Relative yield-map perturbation used by correction growth.",
    )
    parser.add_argument(
        "--correction-feature-fraction",
        type=float,
        default=0.2,
        help="Fraction of the dominant signed correction retained for fitting.",
    )
    parser.add_argument("--minimum-objective-improvement", type=float, default=MINIMUM_OBJECTIVE_IMPROVEMENT)
    parser.add_argument(
        "--fixed-basis-trajectory",
        action="store_true",
        help=(
            "Accept every solved BF stage through --max-basis-functions. "
            "Use for fixed-cap objective investigations whose stage "
            "normalisation changes after refinement."
        ),
    )
    parser.add_argument("--refinement-smoothing-points", type=int, default=3, help="Odd uniform-filter width used when selecting the next EGI basis centre.")
    parser.add_argument("--multistart-basis-placement", action="store_true", help="Screen the EGI peak and four 10%%-offset centre seeds before each full basis solve.")
    parser.add_argument("--multistart-offset-fraction", type=float, default=MULTISTART_OFFSET_FRACTION, help="Centre-span fraction used for the four multi-start offsets.")
    parser.add_argument("--multistart-screening-iterations", type=int, default=MULTISTART_SCREENING_ITERATIONS, help="Pattern-search iterations allocated to each centre candidate.")
    parser.add_argument("--initial-mesh-size", type=float, default=INITIAL_MESH_SIZE, help="Initial normalised pattern-search mesh size.")
    parser.add_argument("--centre-bounds-span-factor", type=float, default=1.0, help="Multiplier applied to the coordinate span used for Gaussian centre bounds.")
    parser.add_argument("--fix-hardening", action="store_true", help="Hold the supplied initial hardening modulus fixed in both phases.")
    parser.add_argument("--sensitivity-spatial-weighting", action="store_true", help="Use frozen phase-start sensitivity-derived EGI and FRE spatial weights.")
    parser.add_argument("--sensitivity-perturbation-factor", type=float, default=SENSITIVITY_PERTURBATION_FACTOR, help="Relative constitutive-parameter perturbation used to calculate spatial weights.")
    parser.add_argument("--sensitivity-weight-floor", type=float, default=SENSITIVITY_WEIGHT_FLOOR, help="Nonzero activity-weight floor before normalisation.")
    parser.add_argument("--minimum-mesh-size", type=float, default=MINIMUM_MESH_SIZE)
    parser.add_argument(
        "--objective-relative-tolerance",
        type=float,
        default=OBJECTIVE_RELATIVE_TOLERANCE,
        help=(
            "Minimum scale-relative objective reduction required to accept a "
            "pattern-search candidate."
        ),
    )
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--max-evaluations", type=int, default=PHASE_1_MAX_EVALUATIONS, help="Maximum pattern-search evaluations in phase 1.")
    parser.add_argument("--parallel-workers", type=int, default=PARALLEL_WORKERS)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED, help="Seed for reproducible pattern-search direction bases.")
    parser.add_argument("--stress-backend", choices=("numpy", "cython"), default=STRESS_BACKEND, help="Stress-reconstruction backend; overrides STRESS_BACKEND.")
    parser.add_argument(
        "--artificial-noise-model",
        type=Path,
        default=None,
        help="Compact WDBN1 noise-model YAML/JSON applied in memory.",
    )
    parser.add_argument(
        "--artificial-noise-scale",
        type=float,
        default=0.0,
        help="Multiplier for the compact strain/force noise model.",
    )
    parser.add_argument("--artificial-noise-seed", type=int, default=20260828)
    parser.add_argument("--no-progress", action="store_false", dest="show_progress", default=SHOW_PROGRESS, help="Disable console progress messages during identification.")
    args = parser.parse_args()
    objective_configs = (
        args.objective_config,
        args.data_driven_objective_config,
        args.simple_data_driven_objective_config,
    )
    if sum(value is not None for value in objective_configs) > 1:
        parser.error("Objective configuration options are mutually exclusive.")
    if not 0.0 < args.initial_mesh_size <= 1.0:
        parser.error("--initial-mesh-size must lie in (0, 1].")
    if not 0.0 < args.minimum_mesh_size <= args.initial_mesh_size:
        parser.error("--minimum-mesh-size must lie in (0, initial mesh size].")
    if (
        not np.isfinite(args.objective_relative_tolerance)
        or not 0.0 <= args.objective_relative_tolerance < 1.0
    ):
        parser.error("--objective-relative-tolerance must lie in [0, 1).")
    if args.max_iterations < 1 or args.max_evaluations < 1 or args.parallel_workers < 1 or args.phase_0_max_evaluations < 1:
        parser.error("Iteration, evaluation, and worker counts must be positive.")
    if not 0.0 <= args.force_weight <= 1.0:
        parser.error("--force-weight must lie in [0, 1].")
    if args.force_slices < 2 or args.max_basis_functions < 1:
        parser.error("Force slices must be at least two and max bases must be positive.")
    if args.egi_window_weights is not None and (
        len(args.egi_window_weights) != len(args.egi_windows)
        or any(weight <= 0.0 for weight in args.egi_window_weights)
    ):
        parser.error("--egi-window-weights must be positive and match --egi-windows.")
    if args.refinement_smoothing_points < 1 or args.refinement_smoothing_points % 2 == 0:
        parser.error("--refinement-smoothing-points must be a positive odd integer.")
    if not 0.0 < args.multistart_offset_fraction <= 1.0:
        parser.error("--multistart-offset-fraction must lie in (0, 1].")
    if args.multistart_screening_iterations < 1:
        parser.error("--multistart-screening-iterations must be positive.")
    if not 0.0 <= args.minimum_objective_improvement < 1.0:
        parser.error("--minimum-objective-improvement must lie in [0, 1).")
    if not np.isfinite(args.centre_bounds_span_factor) or args.centre_bounds_span_factor < 1.0:
        parser.error("--centre-bounds-span-factor must be finite and at least 1.")
    if not 0.0 < args.sensitivity_perturbation_factor < 1.0:
        parser.error("--sensitivity-perturbation-factor must lie in (0, 1).")
    if not 0.0 < args.sensitivity_weight_floor <= 1.0:
        parser.error("--sensitivity-weight-floor must lie in (0, 1].")
    if not 0.0 < args.correction_sensitivity_perturbation_factor < 1.0:
        parser.error(
            "--correction-sensitivity-perturbation-factor must lie in (0, 1)."
        )
    if not 0.0 < args.correction_feature_fraction <= 1.0:
        parser.error("--correction-feature-fraction must lie in (0, 1].")
    if not np.isfinite(args.artificial_noise_scale) or args.artificial_noise_scale < 0.0:
        parser.error("--artificial-noise-scale must be finite and non-negative.")
    if args.artificial_noise_scale > 0.0 and args.artificial_noise_model is None:
        parser.error("A positive artificial-noise scale requires --artificial-noise-model.")
    return args


def _create_phase_1_objective(config_path, global_objective):
    if config_path is None:
        return global_objective
    payload = json.loads(config_path.expanduser().resolve().read_text(encoding="utf-8"))
    terms = []
    references = {}
    for item in payload.get("features", []):
        name = str(item["name"])
        terms.append(MaterialFeatureTerm(
            name=name,
            metric_result_index=int(item["metric_result_index"]),
            reduction=MaterialFeatureReduction(item["reduction"]),
            frame_indices=(
                None if item.get("frame_indices") is None
                else tuple(int(value) for value in item["frame_indices"])
            ),
            weight=float(item.get("weight", 1.0)),
            quantile=float(item.get("quantile", 0.90)),
            sigma_pixels=item.get("sigma_pixels", 2.0),
            spatial_axes=tuple(item.get("spatial_axes", (-2, -1))),
        ))
        floor = float(item.get("noise_floor", 0.0))
        # The stage reference is refreshed immediately before every fixed-BF
        # solve. This placeholder only carries the frozen propagated floor.
        references[name] = MaterialFeatureReference(
            noise_floor=floor,
            stage_reference=floor + max(1.0, abs(floor)),
        )
    return MaterialInformationObjective(
        global_objective=global_objective,
        feature_terms=terms,
        alpha=float(payload.get("alpha", 0.5)),
        smooth_max_temperature=float(payload.get("smooth_max_temperature", 0.1)),
        mean_fraction=float(payload.get("mean_fraction", 0.1)),
        positive_part_temperature=float(payload.get("positive_part_temperature", 1e-3)),
        references=references,
    )


def _create_data_driven_phase_1_objective(
    payload: dict[str, object],
    *,
    timestep_count: int,
    basis_growth_objective,
    diagnostic_callback,
) -> tuple[SensitivityInformationObjective, AutomaticEgiSupportPreparation]:
    """Create the narrow v1 data-driven objective from a durable JSON file.

    All observations are retained in v1 (``load_regime='all'``); the frozen
    native-DOF sensitivity determines which spatial/temporal combinations are
    informative.  Noise scales are explicitly scalar diagonal approximations
    until a campaign provides propagated residual arrays.
    """

    force_noise = float(payload.get("force_noise_scale", 1.0))
    egi_noise = float(payload.get("egi_noise_scale", 1.0))
    blocks = (
        ResidualBlockSpec(
            "fre", 0, "all", "fre", role="fre_guard", noise_scale=force_noise,
        ),
        ResidualBlockSpec(
            "egi_fine", 1, "all", "egi", role="training", noise_scale=egi_noise,
        ),
        ResidualBlockSpec(
            "egi_middle", 2, "all", "egi", role="training", noise_scale=egi_noise,
        ),
        ResidualBlockSpec(
            "egi_broad", 3, "all", "egi", role="broad_egi_guard", noise_scale=egi_noise,
        ),
    )
    objective = SensitivityInformationObjective(
        SensitivityInformationObjectiveConfig(
            # Regimes are present for the canonical residual contract. Every
            # v1 block is 'all', so this neutral placeholder is never used to
            # discard a frame.
            load_regimes=resolve_load_regimes(np.zeros(timestep_count)),
            residual_blocks=blocks,
            finite_difference_step=float(payload.get("finite_difference_step", 1.0e-3)),
            meaningful_dof_movement=float(payload.get("meaningful_dof_movement", 1.0e-2)),
            minimum_noise_response=float(payload.get("minimum_noise_response", 1.0)),
            projection_covariance_floor=float(payload.get("projection_covariance_floor", 1.0e-12)),
            robust_transition=float(payload.get("robust_transition", 1.5)),
        ),
        diagnostic_callback=diagnostic_callback,
        basis_growth_objective=basis_growth_objective,
    )
    preparation = AutomaticEgiSupportPreparation(
        yield_parameter_name=str(payload.get("yield_parameter_name", "yield_strength")),
        yield_parameter_range=float(payload.get("yield_parameter_range", YIELD_BOUNDS_MPA[1] - YIELD_BOUNDS_MPA[0])),
        perturbation_fraction=float(payload.get("probe_perturbation_fraction", 0.01)),
        local_probe_count=int(payload.get("local_probe_count", 9)),
        local_probe_width=(
            None
            if payload.get("local_probe_width") is None
            else float(payload["local_probe_width"])
        ),
        residual_noise_scale=egi_noise,
        bank_config=EgiSupportBankConfig(
            candidate_count=int(payload.get("candidate_count", 10)),
            minimum_pixels=int(payload.get("minimum_pixels", 3)),
            maximum_bbox_fraction=float(payload.get("maximum_bbox_fraction", 0.5)),
        ),
        selection_config=EgiSupportInformationSelectionConfig(
            minimum_coverage_fraction=float(payload.get("minimum_coverage_fraction", 0.5)),
            minimum_response_to_noise=float(payload.get("minimum_response_to_noise", 1.0)),
            minimum_local_probe_fraction=float(payload.get("minimum_local_probe_fraction", 0.5)),
            fisher_regularisation=float(payload.get("fisher_regularisation", 1.0e-12)),
            minimum_middle_information_gain=float(payload.get("minimum_middle_information_gain", 0.0)),
        ),
        diagnostic_callback=diagnostic_callback,
    )
    return objective, preparation


def _create_simple_data_driven_phase_1_objective(
    payload: dict[str, object],
    *,
    x: np.ndarray,
    y: np.ndarray,
    basis_growth_objective,
    diagnostic_callback,
) -> tuple[
    SensitivityGatedEgiObjective,
    SimpleEgiSupportPreparation | FixedEgiSupportPreparation,
]:
    """Create the deliberately minimalist direct-SNR/two-perturbation path."""

    egi_noise_value = payload.get("egi_noise_scales", payload.get("egi_noise_scale", 1.0))
    if isinstance(egi_noise_value, (int, float)):
        egi_noise_scales = (float(egi_noise_value),) * 3
    else:
        egi_noise_scales = tuple(float(value) for value in egi_noise_value)
    force_weight = float(payload.get("force_weight", 0.15))
    broad_guard_weight = float(payload.get("broad_guard_weight", 0.10))
    implied_informative_weight = 1.0 - force_weight - broad_guard_weight
    declared_informative_weight = payload.get("informative_egi_weight")
    if declared_informative_weight is not None and not np.isclose(
        float(declared_informative_weight), implied_informative_weight,
        rtol=0.0, atol=1.0e-12,
    ):
        raise ValueError(
            "informative_egi_weight must equal 1 - force_weight - "
            "broad_guard_weight."
        )
    objective = SensitivityGatedEgiObjective(
        SensitivityGatedObjectiveConfig(
            parameter_names=tuple(payload.get(
                "sensitivity_parameter_names",
                ("yield_strength", "hardening_modulus"),
            )),
            perturbation_factor=float(payload.get("perturbation_factor", 0.01)),
            sensitivity_scaling_percentile=float(payload.get("sensitivity_scaling_percentile", 95.0)),
            gate_start=float(payload.get("gate_start", 0.05)),
            gate_full=float(payload.get("gate_full", 0.30)),
            gate_start_quantile=(
                None if payload.get("gate_start_quantile") is None
                else float(payload["gate_start_quantile"])
            ),
            gate_full_quantile=(
                None if payload.get("gate_full_quantile") is None
                else float(payload["gate_full_quantile"])
            ),
            positive_activity_floor=float(payload.get("positive_activity_floor", 1.0e-6)),
            egi_noise_scales=egi_noise_scales,
            force_noise_scale=float(payload.get("force_noise_scale", 1.0)),
            force_weight=force_weight,
            broad_guard_weight=broad_guard_weight,
            aggregation=str(payload.get("aggregation", "weighted_sum")),
            force_guard_limit=(
                None if payload.get("force_guard_limit") is None
                else float(payload["force_guard_limit"])
            ),
            broad_guard_limit=(
                None if payload.get("broad_guard_limit") is None
                else float(payload["broad_guard_limit"])
            ),
            lexicographic_tie_breaker=float(
                payload.get("lexicographic_tie_breaker", 1.0e-6)
            ),
            refresh_every_solves=(
                None
                if payload.get("refresh_every_solves") is None
                else int(payload["refresh_every_solves"])
            ),
        ),
        diagnostic_callback=diagnostic_callback,
        basis_growth_objective=basis_growth_objective,
    )
    fixed_lengths = payload.get("fixed_egi_side_lengths_mm")
    if fixed_lengths is None:
        preparation = SimpleEgiSupportPreparation(
            residual_noise_scale=float(egi_noise_scales[0]),
            bank_config=EgiSupportBankConfig(
                candidate_count=int(payload.get("candidate_count", 10)),
                minimum_pixels=int(payload.get("minimum_pixels", 3)),
                maximum_bbox_fraction=float(payload.get("maximum_bbox_fraction", 0.5)),
            ),
            selection_config=EgiSignalSelectionConfig(
                minimum_coverage_fraction=float(payload.get("minimum_coverage_fraction", 0.5)),
                minimum_signal_to_noise=float(payload.get("minimum_signal_to_noise", 1.0)),
                active_fraction=float(payload.get("active_fraction", 0.2)),
            ),
            diagnostic_callback=diagnostic_callback,
        )
    else:
        lengths = tuple(float(value) for value in fixed_lengths)
        if len(lengths) != 3 or any(not np.isfinite(value) or value <= 0.0 for value in lengths):
            raise ValueError(
                "fixed_egi_side_lengths_mm must contain exactly three positive finite lengths."
            )
        supports = resolve_physical_egi_supports(
            lengths,
            x,
            y,
            minimum_pixels=int(payload.get("minimum_pixels", 3)),
        )
        if len(supports) != 3:
            raise ValueError(
                "fixed_egi_side_lengths_mm resolves to duplicate pixel supports on this grid."
            )
        preparation = FixedEgiSupportPreparation(
            tuple(zip(("fine", "middle", "broad"), supports, strict=True))
        )
    return objective, preparation


def _parse_egi_windows(value: str) -> tuple[tuple[int, int], ...]:
    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not sizes or any(size < 3 or size % 2 == 0 for size in sizes):
        raise argparse.ArgumentTypeError("EGI windows must be odd integers of at least 3.")
    return tuple((size, size) for size in sizes)


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(not np.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated finite values.")
    return values


def _apply_artificial_noise(experiment_data, args) -> dict[str, object]:
    """Apply reproducible correlated WDBN1-like noise without saving inputs."""
    if args.artificial_noise_scale == 0.0:
        return {"enabled": False, "scale": 0.0, "seed": args.artificial_noise_seed}

    import yaml

    model_path = args.artificial_noise_model.expanduser().resolve()
    model = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    component_names = ("exx", "eyy", "exy")
    rng = np.random.default_rng(args.artificial_noise_seed)
    strain = np.asarray(experiment_data.strain, dtype=np.float64).copy()
    specimen_mask = np.all(np.isfinite(strain), axis=(0, 1))
    spacing = model["grid_spacing_mm"]
    realised: dict[str, float] = {}
    for component_index, name in enumerate(component_names):
        component = model["components"][name]
        sigma = float(component["sigma"]) * args.artificial_noise_scale
        smooth = component["gaussian_filter_sigma_mm"]
        sigma_pixels = (
            float(smooth["y"]) / float(spacing["y"]),
            float(smooth["x"]) / float(spacing["x"]),
        )
        for timestep in range(strain.shape[0]):
            sample = gaussian_filter(
                rng.standard_normal(strain.shape[2:]), sigma=sigma_pixels,
                mode="reflect",
            )
            sample -= float(np.mean(sample[specimen_mask]))
            sample_std = float(np.std(sample[specimen_mask]))
            if sample_std <= np.finfo(float).eps:
                raise RuntimeError("Artificial-noise sample has zero variance.")
            strain[timestep, component_index, specimen_mask] += (
                sigma * sample[specimen_mask] / sample_std
            )
        realised[name] = sigma
    experiment_data.strain = strain

    force_sigma = float(model["force"]["sigma_n"]) * args.artificial_noise_scale
    force = np.asarray(experiment_data.boundary_conditions.force, dtype=np.float64).copy()
    if force.ndim == 1:
        force += rng.normal(0.0, force_sigma, size=force.shape)
    else:
        force[:, 0] += rng.normal(0.0, force_sigma, size=force.shape[0])
    experiment_data.boundary_conditions.force = force
    return {
        "enabled": True,
        "model": str(model_path),
        "scale": args.artificial_noise_scale,
        "seed": args.artificial_noise_seed,
        "strain_sigma": realised,
        "force_sigma_n": force_sigma,
        "correlated_spatially": True,
    }


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
