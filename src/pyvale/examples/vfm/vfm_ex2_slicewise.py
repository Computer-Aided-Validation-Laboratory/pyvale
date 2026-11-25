# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Parallel slicewise identification and result analysis
=====================================================

Identify a four-region yield-strength distribution with independent parallel
slice solves. Then apply the same supported postprocessing pipeline used for
saved experimental results: stress verification, plasticity, force
reconstruction, equilibrium-gap, and parameter-error diagnostics.
"""

from pathlib import Path
from pprint import pprint

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
# Optional saved output
# ---------------------
save_results_file = False
save_figures = False
saved_data_directory = Path.cwd() / "vfm_ex2_slicewise_saved_data"


# %%
# Build a heterogeneous tensile experiment
# -----------------------------------------
# Only yield strength varies between the four longitudinal regions. Elastic
# modulus, Poisson's ratio, and hardening modulus are treated as known.
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
# Configure independent slicewise solves
# --------------------------------------
# The metric and yield-strength parameterisation share the same support object.
# Two worker threads solve independent slices concurrently.
support = vfm.SupportSlice(
    slice_config=vfm.SliceConfig(axis="x", num_slices=4),
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
    optimiser=vfm.SliceWiseIndependentLeastSquares(parallel_workers=2),
)
constitutive_law = vfm.IsotropicVonMisesElastoplasticity(
    vfm.HardeningLinear(),
)
result: vfm.IdentificationResult = vfm.run_identification(
    experiment_data,
    vfm.IdentificationConfig(
        constitutive_law=constitutive_law,
        parameters=parameters,
        phases=[phase],
    ),
)

if save_results_file:
    result_file: Path = result.save_to_yaml(
        saved_data_directory / "identification_result"
    )
    print(f"Saved identification result: {result_file}")

# %%
# Analyse the identification result
# ---------------------------------
# These functions also support results reloaded from an on-disk result bundle.
# EGI remains a diagnostic here; it is not used as an identification objective.
stress: npt.NDArray[np.float64] = vfm.postprocessing.compute_stress_from_result(
    experiment_data,
    result,
    constitutive_law,
)
stress_check: vfm.postprocessing.StressCheckResult = (
    vfm.postprocessing.check_stress_against_saved(stress, result.final_stress)
)
plasticity: vfm.postprocessing.PlasticityDiagnostics | None = (
    vfm.postprocessing.compute_plasticity_diagnostics(
        experiment_data,
        constitutive_law,
        result.parameter_maps,
    )
)
force_reconstruction: vfm.postprocessing.ForceReconstructionDiagnostics = (
    vfm.postprocessing.compute_force_reconstruction_diagnostics(
        experiment_data,
        stress,
        axis="x",
        num_slices=4,
    )
)
equilibrium_gap: vfm.postprocessing.EquilibriumGapDiagnostics = (
    vfm.postprocessing.compute_equilibrium_gap_diagnostics(
        experiment_data,
        stress,
        window_size=3,
    )
)
parameter_errors: vfm.postprocessing.ParameterErrorDiagnostics = (
    vfm.postprocessing.compute_parameter_error_diagnostics(
        result.parameter_maps,
        synthetic_case.known_parameter_maps,
    )
)

summary: dict[str, object] = {
    **stress_check.to_summary(),
    **vfm.postprocessing.parameter_map_summary(result.parameter_maps),
    **force_reconstruction.to_summary(),
    **equilibrium_gap.to_summary(),
    **parameter_errors.summary,
}
if plasticity is not None:
    summary.update(plasticity.to_summary())
pprint(summary, sort_dicts=False)

solve_result: vfm.SolveResult = result.history.phases[0].solve_results[-1]
print(
    "parallel solve: "
    f"workers={solve_result.details['parallel_workers']}, "
    f"slice solves={len(solve_result.children)}"
)

# %%
# Compare the identified and reference yield maps
# -----------------------------------------------
identified_yield: npt.NDArray[np.float64] = result.parameter_maps["yield_strength"]
reference_yield: npt.NDArray[np.float64] = (
    synthetic_case.known_parameter_maps["yield_strength"]
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
    (reference_yield, "Reference yield strength", "viridis"),
    (identified_yield, "Identified yield strength", "viridis"),
    (yield_error_percent, "Identification error [%]", "RdBu_r"),
)
for axis, (values, title, colour_map) in zip(axes, plots, strict=True):
    image = axis.imshow(values, origin="lower", aspect="auto", cmap=colour_map)
    axis.set_title(title)
    axis.set_xlabel("x grid index")
    axis.set_ylabel("y grid index")
    figure.colorbar(image, ax=axis)

egi_figure, egi_axis = plt.subplots(figsize=(6.0, 3.2), constrained_layout=True)
egi_image = egi_axis.imshow(
    equilibrium_gap.weighted_temporal_rms_map,
    origin="lower",
    aspect="auto",
    cmap="viridis",
)
egi_axis.set_title("Equilibrium-gap diagnostic")
egi_axis.set_xlabel("x grid index")
egi_axis.set_ylabel("y grid index")
egi_colourbar = egi_figure.colorbar(egi_image, ax=egi_axis)
egi_colourbar.set_label("Normalised equilibrium gap [-]")

if save_figures:
    figure_directory = saved_data_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    figure_file = figure_directory / "yield_strength_comparison.png"
    figure.savefig(figure_file, dpi=200)
    print(f"Saved figure: {figure_file}")
    egi_figure_file = figure_directory / "normalised_equilibrium_gap.png"
    egi_figure.savefig(egi_figure_file, dpi=200)
    print(f"Saved figure: {egi_figure_file}")

if "agg" not in plt.get_backend().lower():
    plt.show()
