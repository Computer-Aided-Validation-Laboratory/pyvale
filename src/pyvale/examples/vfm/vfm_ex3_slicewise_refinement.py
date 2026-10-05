# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Example 3: slicewise identification with refinement
===================================================

Start with an intentionally over-refined eight-slice representation. After the
first independent solve, merge neighbouring slices with similar identified
parameters and solve the refined four-slice representation again.
"""

from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

import pyvale.vfm as vfm
from pyvale.examples.vfm.synthetic_rectangular_tensile import (
    build_synthetic_identification_parameters,
    build_synthetic_tensile_case,
)
from pyvale.vfm.postprocessing import compute_parameter_error_diagnostics


# %%
# Optional saved output
# ---------------------
save_results_file = False
save_figures = False
saved_data_directory = Path.cwd() / "vfm_ex3_slicewise_refinement_saved_data"


# %%
# Build the synthetic four-region case
synthetic_case = build_synthetic_tensile_case()
experiment_data = synthetic_case.experiment_data
map_size = np.asarray(
    experiment_data.specimen_geometry.x.shape,
    dtype=np.uint32,
)
parameters = build_synthetic_identification_parameters(
    map_size,
    material=synthetic_case.material,
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
result = vfm.run_identification(
    experiment_data,
    vfm.IdentificationConfig(
        constitutive_law=vfm.IsotropicVonMisesElastoplasticity(
            vfm.HardeningLinear(),
        ),
        parameters=parameters,
        phases=[phase],
    ),
)

if save_results_file:
    result_file = result.save_to_yaml(
        saved_data_directory / "identification_result"
    )
    print(f"Saved identification result: {result_file}")

phase_result = result.history.phases[0]
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

yield_snapshot = phase_result.spatial_parameterisations["yield_strength"][0]
print("final yield-strength support:")
pprint(yield_snapshot.summary, sort_dicts=False)

identified_slice_values = np.unique(
    np.round(result.parameter_maps["yield_strength"], decimals=6)
)
print(f"identified yield strengths [MPa]: {identified_slice_values}")

# %%
# Compare the refined result with the reference map
identified_yield = result.parameter_maps["yield_strength"]
reference_yield = synthetic_case.known_parameter_maps["yield_strength"]
parameter_errors = compute_parameter_error_diagnostics(
    result.parameter_maps,
    synthetic_case.known_parameter_maps,
)
yield_error_percent = parameter_errors.percent_error_maps["yield_strength"]

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

if save_figures:
    figure_directory = saved_data_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    figure_file = figure_directory / "refined_yield_strength_comparison.png"
    figure.savefig(figure_file, dpi=200)
    print(f"Saved figure: {figure_file}")

if "agg" not in plt.get_backend().lower():
    plt.show()
