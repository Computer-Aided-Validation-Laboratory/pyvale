"""Run independent slice-wise identification for the vertical-slice weld case.

This file is deliberately configured as a runnable/debuggable VS Code script:
open it, select the PyVale environment, and press Run or Debug.  Both yield
strength and hardening modulus are identified independently in each physical
x-slice.  Command-line arguments remain available for quick experiments.
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

DATA_ROOT = Path(
    "/home/robh/1_Projects/pyvale-vfm-test-data/notched-weld-vertical-slice"
)
INPUT_PATH = DATA_ROOT / "prepared"
OUTPUT_ROOT = DATA_ROOT / "identification"

SLICE_AXIS = "x"
SLICE_BOUNDARIES_MM = (12.5, 35.0, 40.0, 62.5)
SHOW_PROGRESS = True

# The FE truth is E=190 GPa, nu=0.28, yield=360/420/360 MPa and H=3700 MPa.
# E and nu are known here; yield and H are the two slice-wise unknown fields.
ELASTIC_MODULUS_MPA = 190_000.0
POISSONS_RATIO = 0.28
YIELD_INITIAL_MPA = 390.0
YIELD_BOUNDS_MPA = (300.0, 500.0)
HARDENING_INITIAL_MPA = 3_700.0
HARDENING_BOUNDS_MPA = (1_000.0, 8_000.0)


def main() -> None:
    args = _parse_args()
    experiment_data_file = (
        args.input / "experiment_data.yaml"
        if args.input.is_dir()
        else args.input
    )
    experiment_data = ExperimentData.load_from_file(experiment_data_file)

    parameter_map_size = np.asarray(
        experiment_data.specimen_geometry.x.shape,
        dtype=np.uint32,
    )
    parameters = {
        "elastic_modulus": ConstitutiveParameter(
            ELASTIC_MODULUS_MPA,
            150_000.0,
            250_000.0,
            parameter_map_size,
        ),
        "poissons_ratio": ConstitutiveParameter(
            POISSONS_RATIO,
            0.2,
            0.4,
            parameter_map_size,
        ),
        "yield_strength": ConstitutiveParameter(
            YIELD_INITIAL_MPA,
            YIELD_BOUNDS_MPA[0],
            YIELD_BOUNDS_MPA[1],
            parameter_map_size,
        ),
        "hardening_modulus": ConstitutiveParameter(
            HARDENING_INITIAL_MPA,
            HARDENING_BOUNDS_MPA[0],
            HARDENING_BOUNDS_MPA[1],
            parameter_map_size,
        ),
    }

    shared_support = SupportSlice(
        slice_config=SliceConfig(
            axis=args.axis,
            boundaries=np.asarray(args.boundaries, dtype=np.float64),
        ),
    )

    phase = IdentificationPhase(
        spatial_parameterisations={
            "elastic_modulus": [SpatialParameterisationKnown()],
            "poissons_ratio": [SpatialParameterisationKnown()],
            "yield_strength": [
                SliceWiseSpatialParameterisation(support=shared_support)
            ],
            "hardening_modulus": [
                SliceWiseSpatialParameterisation(support=shared_support)
            ],
        },
        metrics=[SliceWiseForceReconstructionMetric(support=shared_support)],
        objective_function=VectorWeightedObjective(),
        optimiser=SliceWiseIndependentLeastSquares(),
        refinement_policy=None,
    )

    result = run_identification(
        experiment_data,
        IdentificationConfig(
            constitutive_law=IsotropicVonMisesElastoplasticity(HardeningLinear()),
            parameters=parameters,
            phases=[phase],
        ),
        input_source=experiment_data_file,
        progress_callback=(
            ConsoleProgressReporter().report if args.show_progress else None
        ),
    )

    output_dir = args.output_root / "notched-weld-vertical-slice"
    result_file = result.save_to_yaml(output_dir)
    summary = {
        "input": str(experiment_data_file),
        "result": str(result_file),
        "axis": args.axis,
        "slice_boundaries_mm": list(args.boundaries),
        "identified_parameters": ["yield_strength", "hardening_modulus"],
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved identification result bundle to {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--axis", choices=("x", "y"), default=SLICE_AXIS)
    parser.add_argument(
        "--boundaries",
        nargs="+",
        type=float,
        default=SLICE_BOUNDARIES_MM,
        help="Physical slice boundaries in mm.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="show_progress",
        default=SHOW_PROGRESS,
        help="Disable console progress messages during identification.",
    )
    args = parser.parse_args()
    if len(args.boundaries) < 2 or any(
        right <= left
        for left, right in zip(args.boundaries, args.boundaries[1:])
    ):
        parser.error("--boundaries must contain at least two strictly increasing values.")
    return args


if __name__ == "__main__":
    main()
