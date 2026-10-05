# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Example 4: homogeneous SBVF followed by slicewise FRE
=====================================================

Run two identification phases in sequence with elastic modulus and Poisson's
ratio fixed at known values. The first identifies homogeneous yield strength
and hardening modulus with SBVF. The second jointly refines homogeneous
hardening modulus and a four-slice yield-strength representation using force
reconstruction error.
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
saved_data_directory = Path.cwd() / "vfm_ex4_hom_slicewise_saved_data"


# %%
# Build one experiment and initial parameter set
synthetic_case = build_synthetic_tensile_case()
experiment_data = synthetic_case.experiment_data
map_size = np.asarray(
    experiment_data.specimen_geometry.x.shape,
    dtype=np.uint32,
)
parameters = build_synthetic_identification_parameters(
    map_size,
    material=synthetic_case.material,
    initial_yield_strength=250.0,
)

# %%
# Phase 0: homogeneous yield strength and hardening modulus
homogeneous_phase = vfm.IdentificationPhase(
    spatial_parameterisations={
        "elastic_modulus": [vfm.SpatialParameterisationKnown()],
        "poissons_ratio": [vfm.SpatialParameterisationKnown()],
        "yield_strength": [vfm.SpatialParameterisationHomogeneous()],
        "hardening_modulus": [vfm.SpatialParameterisationHomogeneous()],
    },
    metrics=[vfm.MetricSBVF(np.array([2, 3], dtype=np.uint32))],
    objective_function=vfm.VectorFirstResultPassthrough(),
    optimiser=vfm.OptimiserLeastSquares(max_evaluations=100),
    optimisation_newton_tolerance=1.0e-8,
)

# %%
# Phase 1: slicewise FRE
# ----------------------
# Values identified in phase 0 become the initial parameter maps for phase 1.
# Elastic modulus and Poisson's ratio remain known. Yield strength gains one
# active degree of freedom per slice while hardening modulus remains an active
# homogeneous parameter shared by every slice.
support = vfm.SupportSlice(
    slice_config=vfm.SliceConfig(axis="x", num_slices=4),
)
slicewise_phase = vfm.IdentificationPhase(
    spatial_parameterisations={
        "elastic_modulus": [vfm.SpatialParameterisationKnown()],
        "poissons_ratio": [vfm.SpatialParameterisationKnown()],
        "yield_strength": [
            vfm.SliceWiseSpatialParameterisation(support=support)
        ],
        "hardening_modulus": [vfm.SpatialParameterisationHomogeneous()],
    },
    metrics=[vfm.SliceWiseForceReconstructionMetric(support=support)],
    objective_function=vfm.VectorWeightedObjective(),
    optimiser=vfm.OptimiserLeastSquares(max_evaluations=100),
    optimisation_newton_tolerance=1.0e-8,
)

# %%
# Run both phases and compare with the reference maps
result = vfm.run_identification(
    experiment_data,
    vfm.IdentificationConfig(
        constitutive_law=vfm.IsotropicVonMisesElastoplasticity(
            vfm.HardeningLinear(),
            error_tolerance=1.0e-10,
        ),
        parameters=parameters,
        phases=[homogeneous_phase, slicewise_phase],
    ),
)

if save_results_file:
    result_file = result.save_to_yaml(
        saved_data_directory / "identification_result"
    )
    print(f"Saved identification result: {result_file}")

phase_zero_snapshot = result.history.phases[0].spatial_parameterisations
phase_zero_values = {
    name: phase_zero_snapshot[name][0].summary["value"]
    for name in ("yield_strength", "hardening_modulus")
}
print("phase 0 homogeneous estimates:")
pprint(phase_zero_values, sort_dicts=False)

errors = compute_parameter_error_diagnostics(
    result.parameter_maps,
    synthetic_case.known_parameter_maps,
)
print("phase 1 final parameter errors:")
pprint(errors.summary, sort_dicts=False)
print(f"completed phases: {len(result.history.phases)}")

# %%
# Compare the homogeneous and slicewise yield-strength results
reference_yield = synthetic_case.known_parameter_maps["yield_strength"]
phase_zero_yield = np.full_like(
    reference_yield,
    phase_zero_values["yield_strength"],
)
phase_one_yield = result.parameter_maps["yield_strength"]
yield_error_percent = errors.percent_error_maps["yield_strength"]

figure, axes = plt.subplots(
    2,
    2,
    figsize=(10.0, 6.4),
    constrained_layout=True,
)
plots = (
    (reference_yield, "Reference yield strength [MPa]", "viridis"),
    (phase_zero_yield, "Phase 0: homogeneous [MPa]", "viridis"),
    (phase_one_yield, "Phase 1: slicewise [MPa]", "viridis"),
    (yield_error_percent, "Final identification error [%]", "RdBu_r"),
)
for axis, (values, title, colour_map) in zip(axes.ravel(), plots, strict=True):
    image = axis.imshow(values, origin="lower", aspect="auto", cmap=colour_map)
    axis.set_title(title)
    axis.set_xlabel("x grid index")
    axis.set_ylabel("y grid index")
    figure.colorbar(image, ax=axis)

if save_figures:
    figure_directory = saved_data_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    figure_file = figure_directory / "yield_strength_phase_comparison.png"
    figure.savefig(figure_file, dpi=200)
    print(f"Saved figure: {figure_file}")

if "agg" not in plt.get_backend().lower():
    plt.show()
