# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Slicewise identification with refinement
========================================

Start with an intentionally over-refined eight-slice representation. After the
first independent solve, merge neighbouring slices with similar identified
parameters and solve the refined four-slice representation again.
"""

from pathlib import Path
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import pyvale.vfm as vfm
from pyvale.examples.vfm.synthetic_rectangular_tensile import (
    SyntheticTensileCase,
    build_synthetic_identification_parameters,
    build_synthetic_tensile_case,
)


# %%
matplotlib.use("Agg")

SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")

output_path = Path.cwd() / "pyvale-output" / "vfm_ex3_slicewise_refinement"
output_path.mkdir(parents=True, exist_ok=True)
SAVE_RESULTS_FILE = False

# %%
# Build the synthetic four-region case
synthetic_case: SyntheticTensileCase = build_synthetic_tensile_case()
experiment_data: vfm.ExperimentData = synthetic_case.experiment_data
map_size: npt.NDArray[np.uint32] = np.asarray(
    experiment_data.specimen_geometry.x.shape,
    dtype=np.uint32,
)
parameters: dict[str, vfm.ConstitutiveParameter] = (
    build_synthetic_identification_parameters(
        map_size,
        material=synthetic_case.material,
    )
)

# %%
# Configure an over-refined support
# ---------------------------------
# The merge tolerance is expressed as a relative parameter difference. The
# force-error threshold is deliberately high so this example demonstrates the
# merge path without introducing additional splits.
support = vfm.SupportSlice(
    slice_config=vfm.SliceConfig(axis="x", num_slices=8),
)
phase = vfm.IdentificationPhase(
    spatial_parameterisations={
        "elastic_modulus": [vfm.SpatialParameterisationKnown()],
        "poissons_ratio": [vfm.SpatialParameterisationKnown()],
        "yield_strength": [
            vfm.SliceWiseSpatialParameterisation(support=support)
        ],
        "hardening_modulus": [vfm.SpatialParameterisationKnown()],
    },
    metrics=[vfm.SliceWiseForceReconstructionMetric(support=support)],
    objective_function=vfm.VectorWeightedObjective(),
    optimiser=vfm.SliceWiseIndependentLeastSquares(),
    refinement_policy=vfm.SliceMergeSplitRefinement(
        target=support,
        merge_parameter_tolerance=0.04,
        split_error_threshold=1.0,
        max_refinements=1,
    ),
)

# %%
# Run and inspect the refinement history
result: vfm.IdentificationResult = vfm.run_identification(
    experiment_data,
    vfm.IdentificationConfig(
        constitutive_law=vfm.IsotropicVonMisesElastoplasticity(
            vfm.HardeningLinear(),
        ),
        parameters=parameters,
        phases=[phase],
    ),
)

if SAVE_RESULTS_FILE:
    result_file: Path = result.save_to_yaml(
        output_path / "identification_result"
    )
    print(f"Saved identification result: {result_file}")

phase_result: vfm.PhaseResult = result.history.phases[0]

print(f"solve count: {len(phase_result.solve_results)}")
print(f"refinement events: {len(phase_result.refinement_events)}")
for event in phase_result.refinement_events:
    pprint(
        {
            "before": event.before_summary,
            "after": event.after_summary,
            "trigger": event.trigger_summary,
        },
        sort_dicts=False,
    )

yield_snapshot: vfm.ParameterisationSnapshot = (
    phase_result.spatial_parameterisations["yield_strength"][0]
)
print("final yield-strength support:")
pprint(yield_snapshot.summary, sort_dicts=False)

identified_slice_values: npt.NDArray[np.float64] = np.unique(
    np.round(result.parameter_maps["yield_strength"], decimals=6)
)
print(f"identified yield strengths [MPa]: {identified_slice_values}")

# %%
# Compare the refined result with the reference map
identified_yield: npt.NDArray[np.float64] = result.parameter_maps["yield_strength"]
reference_yield: npt.NDArray[np.float64] = (
    synthetic_case.known_parameter_maps["yield_strength"]
)
parameter_errors: vfm.postprocessing.ParameterErrorDiagnostics = (
    vfm.postprocessing.compute_parameter_error_diagnostics(
        result.parameter_maps,
        synthetic_case.known_parameter_maps,
    )
)
yield_error_percent: npt.NDArray[np.float64] = (
    parameter_errors.percent_error_maps["yield_strength"]
)

figure, axes = plt.subplots(
    1,
    3,
    figsize=(12.0, 3.2),
    constrained_layout=True,
)
plots = (
    (reference_yield, "Reference yield strength [MPa]", "viridis"),
    (identified_yield, "Refined yield strength [MPa]", "viridis"),
    (yield_error_percent, "Identification error [%]", "RdBu_r"),
)
for axis, (values, title, colour_map) in zip(axes, plots, strict=True):
    image = axis.imshow(values, origin="lower", aspect="auto", cmap=colour_map)
    axis.set_title(title)
    axis.set_xlabel("x grid index")
    axis.set_ylabel("y grid index")
    figure.colorbar(image, ax=axis)

figure_file = output_path / "refined_yield_strength_comparison.png"
figure.savefig(figure_file, dpi=200)
print(f"Saved figure: {figure_file}")

if SHOW_PLOTS:
    plt.show()
else:
    plt.close(figure)
