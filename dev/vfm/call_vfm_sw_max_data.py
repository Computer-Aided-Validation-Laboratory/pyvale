"""Run slice-wise VFM identification for the supplied MAX prepared data.

The DIC fields show axial extension in ``x`` (``eps_xx`` is positive and
``eps_yy`` is lateral contraction), whereas the supplied force array places
the tensile load in its second column.  PyVale's force convention is
``[Fx, Fy]``; this caller therefore maps the measured second column to ``Fx``
in memory before identification.  Source data are never modified.
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


# =============================================================================
# User inputs
# =============================================================================

INPUT_PATH = Path(
    "/home/robh/1_Projects/pyvale-vfm-test-data/max-data/prepared"
)
OUTPUT_ROOT = INPUT_PATH.parent / "call_vfm_sw_max_data_output"

# The axial direction inferred from the strain field is x.  The source force
# array is [0, applied_force], despite ExperimentData requiring [Fx, Fy].
SLICE_AXIS = "x"
SOURCE_FORCE_COMPONENT = 1
NUM_SLICES = 20
MAX_REFINEMENTS = 0
MERGE_PARAMETER_TOLERANCE = 0.05
SPLIT_FORCE_ERROR_THRESHOLD = 0.1
SHOW_PROGRESS = True


def main() -> None:
    args = _parse_args()
    experiment_data_file = (
        args.input / "experiment_data.yaml"
        if args.input.is_dir()
        else args.input
    )
    experiment_data = ExperimentData.load_from_file(experiment_data_file)
    _map_source_force_to_loading_axis(experiment_data, args.axis)

    parameter_map_size = np.asarray(
        experiment_data.specimen_geometry.x.shape,
        dtype=np.uint32,
    )
    parameters = {
        "elastic_modulus": ConstitutiveParameter(
            210_000.0, 150_000.0, 250_000.0, parameter_map_size
        ),
        "poissons_ratio": ConstitutiveParameter(
            0.3, 0.2, 0.4, parameter_map_size
        ),
        "yield_strength": ConstitutiveParameter(
            250.0, 100.0, 2_000.0, parameter_map_size
        ),
        "hardening_modulus": ConstitutiveParameter(
            7_000.0, 1_000.0, 50_000.0, parameter_map_size
        ),
    }

    shared_support = SupportSlice(
        slice_config=SliceConfig(axis=args.axis, num_slices=args.num_slices)
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
        refinement_policy=SliceMergeSplitRefinement(
            target=shared_support,
            merge_parameter_tolerance=args.merge_parameter_tolerance,
            split_error_threshold=args.split_force_error_threshold,
            max_refinements=args.max_refinements,
        ),
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

    output_dir = args.output_root / experiment_data_file.parent.name / "identification_result"
    result_file = result.save_to_yaml(output_dir)
    print(json.dumps({
        "input": str(experiment_data_file),
        "result": str(result_file),
        "axis": args.axis,
        "source_force_component": SOURCE_FORCE_COMPONENT,
        "num_slices": args.num_slices,
        "max_refinements": args.max_refinements,
    }, indent=2))
    print(f"Saved identification result bundle to {output_dir}")


def _map_source_force_to_loading_axis(
    experiment_data: ExperimentData,
    loading_axis: str,
) -> None:
    """Map the source's measured load component into PyVale's [Fx, Fy] order."""
    source_force = np.asarray(experiment_data.boundary_conditions.force)
    if source_force.ndim != 2 or source_force.shape[1] <= SOURCE_FORCE_COMPONENT:
        raise ValueError(
            "Expected the MAX source force array to have at least two columns."
        )
    mapped_force = np.zeros_like(source_force, dtype=np.float64)
    target_component = 0 if loading_axis == "x" else 1
    mapped_force[:, target_component] = source_force[:, SOURCE_FORCE_COMPONENT]
    experiment_data.boundary_conditions.force = mapped_force


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--axis",
        choices=(SLICE_AXIS,),
        default=SLICE_AXIS,
        help="Fixed to x: the axial direction inferred from the DIC strain fields.",
    )
    parser.add_argument("--num-slices", type=int, default=NUM_SLICES)
    parser.add_argument("--max-refinements", type=int, default=MAX_REFINEMENTS)
    parser.add_argument(
        "--merge-parameter-tolerance", type=float,
        default=MERGE_PARAMETER_TOLERANCE,
    )
    parser.add_argument(
        "--split-force-error-threshold", type=float,
        default=SPLIT_FORCE_ERROR_THRESHOLD,
    )
    parser.add_argument(
        "--no-progress", action="store_false", dest="show_progress",
        default=SHOW_PROGRESS, help="Disable console progress messages.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
