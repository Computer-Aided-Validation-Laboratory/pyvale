# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Interpreting validation diagnostics
================================================================================

Interpret conditional CDF families, uncertainty envelopes, directional metric
distributions and MAVM support-shift diagnostics. This standalone example
combines sampled model scenarios, time-dependent data and calibration offsets.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyvale.data as dataset
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.valid as val
from pyvale.valid import valplot, valvis

matplotlib.use("Agg")

# %%
# Console inspection is printed below. Enable interactive windows with SHOW_PLOTS.
SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")
output_path = Path.cwd() / "pyvale-output" / "valid_ex6_diagnostics"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Load and transform data
# -----------------------------------------
# The plate has 12 sensor locations on a 4 by 3 grid: x=10..90 mm,
# y=10..40 mm, z=0. Each site has a thermocouple (one temperature component)
# and an in-plane strain rosette (xx, yy, xy). Six virtual repeat experiments
# supply aleatory samples at five common physical times.
# The second simulation has conductivity and thermal expansion reduced by 10%.
# We treat its noisy measurements as the experiment to expose model discrepancy.
# Prepared arrays already follow the validation axis order declared below.
# Temperature has one component; strain has three components (xx, yy, xy).
with np.load(dataset.valid_data_dir() / "plate_validation_2d.npz") as saved:
    coords = saved["coords"].copy()
    times = saved["times"].copy()
    sim_temp = saved["sim_temp"].copy()  # (12, 1, 5, 2, 6)
    exp_temp = saved["exp_temp"].copy()  # (12, 1, 5, 6)
    sim_strain = saved["sim_strain"].copy()
    exp_strain = saved["exp_strain"].copy()

# %%
# A shared domain registers units, component order and physical coordinates.
# Validation compares corresponding physical cells, retaining distinct point,
# component, time and repeat-experiment axes.
geometry = val.ValGeometry(
    val.EValGeometry.POINT, coords, coord_units="mm", coord_frame="plate"
)
temperature_tag = "plate_temperature"
strain_tag = "plate_strain"
domain = val.ValDomain(
    {
        temperature_tag: val.ValObservable(
            geometry=geometry,
            operator=val.EValOperator.SAMPLE,
            components=("temperature",),
            times=times,
            phys_units="degC",
            time_units="s",
        ),
        strain_tag: val.ValObservable(
            geometry=geometry,
            operator=val.EValOperator.SAMPLE,
            components=("strain_xx", "strain_yy", "strain_xy"),
            times=times,
            phys_units="1",
            time_units="s",
        ),
    }
)

# EPISTEMIC_TRANS holds two alternative model parameter settings: nominal
# then perturbed. Each setting contains its own six aleatory observations.
# Physical axes identify each comparison cell; ALEATORY holds its repeated
# observations. The declared axis order matches the arrays in the archive.
sim_axes = (
    val.EValAxis.POINT,
    val.EValAxis.COMPONENT,
    val.EValAxis.TIME,
    val.EValAxis.EPISTEMIC_TRANS,
    val.EValAxis.ALEATORY,
)
exp_axes = (
    val.EValAxis.POINT,
    val.EValAxis.COMPONENT,
    val.EValAxis.TIME,
    val.EValAxis.ALEATORY,
)
sim_arrays: dict[str, val.ValArray] = {}
exp_arrays: dict[str, val.ValArray] = {}
for observable_tag, sim_values, exp_values in (
    (temperature_tag, sim_temp, exp_temp),
    (strain_tag, sim_strain, exp_strain),
):
    # Lower and upper magnitudes describe a coherent calibration shift in
    # observable units, shared by every repeat at a physical comparison cell.
    magnitude = 0.5 if observable_tag == temperature_tag else 4.0e-6
    if observable_tag == temperature_tag:
        spatial = coords[:, 0, None] / 100.0  # (point, singleton time)
        temporal = times[None, :] / times[-1]  # (singleton point, time)
        bound = val.EpistemicOffsets(
            lower=magnitude * (1.0 + 0.2 * spatial + 0.2 * temporal),
            upper=1.5 * magnitude * (1.0 + 0.1 * spatial + 0.3 * temporal),
            axes=(val.EValAxis.POINT, val.EValAxis.TIME),
        )
    else:
        bound = val.EpistemicOffsets(lower=magnitude, upper=1.5 * magnitude)
    sim_arrays[observable_tag] = val.ValArray(sim_values, sim_axes)
    exp_arrays[observable_tag] = val.ValArray(exp_values, exp_axes, offsets=bound)

val_case = val.ValCase(domain, sim_arrays, exp_arrays)

# %%
# 2. Run validation analysis
# --------------------------
# AVM gives a distance in the observable's units. MAVM additionally separates
# positive and negative support mismatch; INCLUDE retains repeated observations.
mavm = val.MetricMAVM(duplicates=val.EMAVMDuplicates.INCLUDE)
val_result = val.ValidationEngine([val.MetricAVM(), mavm]).run(val_case)

# %%
# 3. Post-processing and inspect data structures
# ----------------------------------------------
# metric_field.axes names each array dimension. The first three axes are always
# (point, component, time), including singleton time for steady comparisons.
# Non-singleton epistemic/offset axes follow in the engine's canonical order.
metric_field = val_result.get(temperature_tag, val.EMetricType.MAVM)
metric_shape = metric_field.metric.shape
metric_axes = metric_field.axes
print("Simulation axes:", sim_arrays[temperature_tag].axes)
print("Simulation shape:", sim_arrays[temperature_tag].data.shape)
print("Result axes:", metric_axes, "shape:", metric_shape)

# %%
# 4. Analysis and visualisation
# ------------------------------
# Use the SensorSim appearance options so fonts, resolution and line styles
# agree with sensor figures. Each CDF panel overlays simulation and experiment;
# dashed curves show the experimental MAVM confidence bounds.
point_idxs = tuple(range(len(coords)))
fig, axes = valplot.plot_val_cdfs_points(
    val_case,
    val_result,
    temperature_tag,
    point_idxs=point_idxs,
    time_idx=4,
    metric=mavm,
    labels=valplot.ValPlotLabels(x_label="Temperature [degC]"),
)
fig.savefig(output_path / "temperature_cdfs.png", bbox_inches="tight")

# %%
# Envelopes describe all retained scenarios at a site. They are pointwise ECDF
# bounds over the conditional distributions. Show every plate sensor.
fig, axes = valplot.plot_val_cdf_envelope_points(
    val_case,
    temperature_tag,
    source=val.EValSource.EXP,
    time_idx=4,
    labels=valplot.ValPlotLabels(x_label="Temperature [degC]"),
)
fig.savefig(output_path / "temperature_envelopes.png", bbox_inches="tight")

# %%
# Each panel follows one physical sensor through time. Explicit selectors fix
# any remaining epistemic scenario; d+ identifies the directional MAVM quantity.
fig, axes = valplot.plot_val_metric_time_points(
    val_result,
    temperature_tag,
    val.EMetricType.MAVM,
    point_idxs=point_idxs,
    selectors={
        val.EValResAxis.SIM_TRANS: 0,
        val.EValResAxis.EXP_OFFSET: val.EEpisOffset.UPPER,
    },
    part=val.EValMetricPart.D_PLUS,
)
fig.savefig(output_path / "temperature_d_plus_time.png", bbox_inches="tight")

# %%
# Overlay validation points on the nominal model in its original xyz geometry.
# The model mesh and sensor positions must share mm coordinates and the plate
# coordinate frame. Background nodal scalars (n_nodes,) have a physical colourbar.
sim_path = dataset.thermomechanical_2d_experiment_paths()[0]
sim_data = mh.ExodusLoader(sim_path).load_all_sim_data()
sens.scale_length_units(1000.0, sim_data, disp_keys=("disp_x", "disp_y"))
background_time_idx = len(sim_data.time) - 1
background_field = valvis.ValSimField(
    sim_data.node_vars["temperature"][:, background_time_idx],
    background_time_idx,
    sens.SensorDescriptor(name="Temperature", symbol=r"T", units="degC"),
)
point_labels = tuple(sens.SensorDescriptor(tag="TC").create_sensor_tags(len(coords)))
# Blue/red encode negative/positive MAVM directional balance. A white point
# can have nonzero total mismatch because d+ and d- can cancel in this display.
pv_plot = valvis.plot_val_metric_on_sim(
    val_result,
    temperature_tag,
    val.EMetricType.MAVM,
    selectors={
        val.EValResAxis.SIM_TRANS: 0,
        val.EValResAxis.TIME: 4,
        val.EValResAxis.EXP_OFFSET: val.EEpisOffset.UPPER,
    },
    part=val.EValMetricPart.BALANCE,
    sim_data=sim_data,
    sim_field=background_field,
    sim_coord_units="mm",
    sim_coord_frame="plate",
    vis_opts=valvis.create_val_vis_opts(interactive=True) if SHOW_PLOTS else None,
    metric_opts=valvis.ValVisOpts(
        point_labels=point_labels, labels_always_visible=True
    ),
)
pv_plot.screenshot(output_path / "temperature_mavm_balance_on_sim.png")
if SHOW_PLOTS:
    pv_plot.show()
pv_plot.close()
# %%
# Figures are saved even when interactive plotting is disabled. Keeping the
# complete analysis in its own directory prevents clashes with other examples.
# %%
# Additional distribution diagnostics
# ------------------------------------
# Inspect every sensor at the same physical time. Whole model scenarios are
# selected by support sum for the endpoint correction; this is distinct from
# the pointwise envelope and from the largest MAVM over all scenarios.
# Calibration offsets, when present, shift an entire conditional sample coherently.
# Start with the conditional samples. extract_aleatory_scenarios accepts
# epis_trans_idx and epis_nuis_idx to inspect a chosen parameter setting; None retains
# every setting. offset=None includes nominal, lower and upper calibration
# shifts. Every returned array is one conditional aleatory sample.
model_quantity = val_case.sim[temperature_tag]
experiment_quantity = val_case.exp[temperature_tag]
# Pointwise envelopes choose extrema at each probability and describe the full
# family spread. SUPPORT_SUM selects whole scenarios by the sum of their unique
# support values; lower_idx and upper_idx identify those original samples.
# Quantile averaging supplies a representative curve through the family.
fig, axes = valplot.create_val_plot_grid(len(coords))
for point_idx, ax in enumerate(axes.flat):
    valplot.plot_val_ecdf_family(
        model_quantity,
        point_idx=point_idx,
        component_idx=0,
        time_idx=4,
        ax=ax,
        labels=valplot.ValPlotLabels(
            title=f"Point {point_idx}: conditional model CDFs",
            x_label="Temperature [degC]",
        ),
    )
fig.savefig(output_path / "conditional_cdf_families.png", bbox_inches="tight")
if not SHOW_PLOTS:
    plt.close(fig)

# Read the four views in order: DISTRIBUTIONS compares model and experiment;
# COMPARISONS shows all four lower/upper endpoint pairings; CORRECTION overlays
# the selected directional support shifts; MODEL_CORRECTION shows the resulting
# model support envelope. The shifts retain the observable's physical units.
# d+ extends the upper model support and d- extends the lower model support.
diagnostic_views = (
    val.EValMAVMDiagnostic.DISTRIBUTIONS,
    val.EValMAVMDiagnostic.COMPARISONS,
    val.EValMAVMDiagnostic.CORRECTION,
    val.EValMAVMDiagnostic.MODEL_CORRECTION,
)
for view in diagnostic_views:
    fig, axes = valplot.create_val_plot_grid(len(coords))
    for point_idx, ax in enumerate(axes.flat):
        model_scenarios = val.extract_aleatory_scenarios(
            model_quantity,
            point_idx=point_idx,
            component_idx=0,
            time_idx=4,
        )
        experimental_scenarios = val.extract_aleatory_scenarios(
            experiment_quantity,
            point_idx=point_idx,
            component_idx=0,
            time_idx=4,
            offset=None,
        )
        model_summary = val.calc_ecdf_summary(
            model_scenarios,
            bounds=val.EValECDFBounds.SUPPORT_SUM,
        )
        experiment_summary = val.calc_ecdf_summary(
            experimental_scenarios,
            bounds=val.EValECDFBounds.SUPPORT_SUM,
        )
        correction = val.calc_mavm_correction(
            model_scenarios[model_summary.lower_idx],
            model_scenarios[model_summary.upper_idx],
            experimental_scenarios[experiment_summary.lower_idx],
            experimental_scenarios[experiment_summary.upper_idx],
        )
        valplot.plot_val_mavm_diagnostic(
            model_summary,
            experiment_summary,
            correction=correction,
            view=view,
            ax=ax,
            labels=valplot.ValPlotLabels(
                title=f"Point {point_idx}: MAVM support shifts",
                x_label="Temperature [degC]",
            ),
        )
    fig.savefig(output_path / f"mavm_{view.name.lower()}.png", bbox_inches="tight")
    if not SHOW_PLOTS:
        plt.close(fig)

# %%
# Histograms count physical/scenario cells within each distance bin. Separate
# directional magnitudes expose asymmetry between d+ and d-.
fig, axes = valplot.create_val_plot_grid(2)
for part, ax in zip(
    (val.EValMetricPart.D_PLUS, val.EValMetricPart.D_MINUS), axes.flat, strict=True
):
    valplot.plot_val_metric_distribution(
        val_result,
        temperature_tag,
        val.EMetricType.MAVM,
        part=part,
        ax=ax,
        labels=valplot.ValPlotLabels(
            title="Temperature: retained physical/scenario cells"
        ),
    )
fig.savefig(output_path / "directional_metric_distributions.png", bbox_inches="tight")
if not SHOW_PLOTS:
    plt.close(fig)

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
