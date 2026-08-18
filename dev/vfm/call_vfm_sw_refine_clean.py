from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyvale.vfm import (
    ConstitutiveParameter,
    ExperimentData,
    HardeningLinear,
    IdentificationConfig,
    IdentificationPhase,
    IsotropicVonMisesElastoplasticity,
    ConsoleProgressReporter,
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


# =============================================================================
# User inputs
# =============================================================================

INPUT_PATH = Path(
    "/media/data/3_Resources/gr91-weld-dic-results/wdbn1/pyvale-input/"
    "vfm-input-data_2026-08-12_15-43"
)
OUTPUT_ROOT = Path(__file__).resolve().parent / "call_vfm_sw_refine_clean_output"

SLICE_AXIS = "y"
NUM_SLICES = 20
MAX_REFINEMENTS = 0
MERGE_PARAMETER_TOLERANCE = 0.05
SPLIT_FORCE_ERROR_THRESHOLD = 0.1
SHOW_PROGRESS = True


def main() -> None:
    # Parse command line arguments
    args = _parse_args()

    # Load data
    experiment_data_file = (
        args.input / "experiment_data.yaml"
        if args.input.is_dir()
        else args.input
    )
    experiment_data = ExperimentData.load_from_file(experiment_data_file)

    # Define constitutive law 
    constitutive_law = IsotropicVonMisesElastoplasticity(HardeningLinear())

    # Define constitutive parameters with bounds and initial guesses
    parameter_map_size = np.asarray(
        experiment_data.specimen_geometry.x.shape,
        dtype=np.uint32,
    )
    parameters = {
        "elastic_modulus": ConstitutiveParameter(
            210_000.0,
            150_000.0,
            250_000.0,
            parameter_map_size,
        ),
        "poissons_ratio": ConstitutiveParameter(
            0.3,
            0.2,
            0.4,
            parameter_map_size,
        ),
        "yield_strength": ConstitutiveParameter(
            250.0,
            100.0,
            2_000.0,
            parameter_map_size,
        ),
        "hardening_modulus": ConstitutiveParameter(
            7_000.0,
            1_000.0,
            50_000.0,
            parameter_map_size,
        ),
    }

    # Define shared support to be used for slice-wise parameterisations and metrics
    shared_support = SupportSlice(
        slice_config=SliceConfig(
            axis=args.axis,
            num_slices=args.num_slices,
        ),
    )

    # Define identification phases 
    phase = IdentificationPhase(
        spatial_parameterisations={
            "elastic_modulus": [SpatialParameterisationKnown()],
            "poissons_ratio": [SpatialParameterisationKnown()],
            "yield_strength": 
                [SliceWiseSpatialParameterisation(support=shared_support)],
            "hardening_modulus":
                [SliceWiseSpatialParameterisation(support=shared_support)],
        },
        metrics=[SliceWiseForceReconstructionMetric(support=shared_support)],
        objective_function=VectorWeightedObjective(),
        optimiser=SliceWiseIndependentLeastSquares(),
        refinement_policy=SliceMergeSplitRefinement(
            target=shared_support,
            merge_parameter_tolerance=args.merge_parameter_tolerance,
            split_error_threshold=args.split_force_error_threshold,
            max_refinements=args.max_refinements,
        ),
    )

    # Gather defined configuration into a single object
    identification_config = IdentificationConfig(
        constitutive_law=constitutive_law,
        parameters=parameters,
        phases=[phase],
    )

    # Run identification
    result = run_identification(
        experiment_data,
        identification_config,
        input_source=experiment_data_file,
        progress_callback=(
            ConsoleProgressReporter().report
            if args.show_progress
            else None
        ),
    )

    # Save results
    output_dir = args.output_root / experiment_data_file.parent.name / "identification_result"
    result_file = result.save_to_yaml(output_dir)

    # Print summary
    summary = {
        "input": str(experiment_data_file),
        "result": str(result_file),
        "axis": args.axis,
        "num_slices": args.num_slices,
        "max_refinements": args.max_refinements,
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved identification result bundle to {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run slice-wise VFM identification and save a clean result bundle."
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--axis", choices=("x", "y"), default=SLICE_AXIS)
    parser.add_argument("--num-slices", type=int, default=NUM_SLICES)
    parser.add_argument("--max-refinements", type=int, default=MAX_REFINEMENTS)
    parser.add_argument(
        "--merge-parameter-tolerance",
        type=float,
        default=MERGE_PARAMETER_TOLERANCE,
    )
    parser.add_argument(
        "--split-force-error-threshold",
        type=float,
        default=SPLIT_FORCE_ERROR_THRESHOLD,
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="show_progress",
        default=SHOW_PROGRESS,
        help="Disable console progress messages during identification.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
