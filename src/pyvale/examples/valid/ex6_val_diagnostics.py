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

matplotlib.use("Agg")

# %%
# Interactive windows and console inspection are opt-in. Change the backend to
# a GUI backend when enabling SHOW_PLOTS outside a notebook.
SHOW_PLOTS = False
VERBOSE = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")
output_path = Path.cwd() / "pyvale-output" / "valid_ex6_diagnostics"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Load and transform data (pre-processing)
# -----------------------------------------
# The plate has 12 sensor locations on a 4 by 3 grid: x=10..90 mm,
# y=10..40 mm, z=0. Each site has a thermocouple (one temperature component)
# and an in-plane strain rosette (xx, yy, xy). Six virtual repeat experiments
# supply aleatory samples at five common physical times.
# The second simulation has conductivity and thermal expansion reduced by 10%.
# We treat its noisy measurements as the experiment to expose model discrepancy.
with np.load(dataset.valid_data_dir() / "plate_validation_2d.npz") as saved:
    coords = saved["coords"].copy()  # (12 points, 3 Cartesian coordinates), mm
    times = saved["times"].copy()  # (5 times,), s
    nominal_temp = saved["nominal_temperature"].copy()  # (6, 12, 1, 5)
    measured_temp = saved["perturbed_temperature"].copy()
    nominal_strain = saved["nominal_strain"].copy()  # (6, 12, 3, 5)
    measured_strain = saved["perturbed_strain"].copy()

# %%
# A shared domain registers units, component order and physical coordinates.
# Validation compares corresponding physical cells, retaining distinct point,
# component, time and repeat-experiment axes.
geometry = val.ValGeometry(
    val.EValGeometry.POINT, coords, coordinate_units="mm", frame="plate"
)
temperature_tag = "plate_temperature"
strain_tag = "plate_strain"
domain = val.ValDomain(
    {
        temperature_tag: val.ValObservable(
            geometry,
            val.EValOperator.SAMPLE,
            ("temperature",),
            times,
            "degC",
            time_units="s",
        ),
        strain_tag: val.ValObservable(
            geometry,
            val.EValOperator.SAMPLE,
            ("strain_xx", "strain_yy", "strain_xy"),
            times,
            "1",
            time_units="s",
        ),
    }
)

sim_arrays: dict[str, val.ValArray] = {}
exp_arrays: dict[str, val.ValArray] = {}
for observable_tag, nominal, measured in (
    (temperature_tag, nominal_temp, measured_temp),
    (strain_tag, nominal_strain, measured_strain),
):
    # Move repeats from axis 0 to the final aleatory axis:
    # (point, component, time, aleatory) = (12, 1 or 3, 5, 6).
    sim_values = np.moveaxis(nominal, 0, -1)
    exp_values = np.moveaxis(measured, 0, -1)
    exp_axes = (
        val.EValAxis.POINT,
        val.EValAxis.COMPONENT,
        val.EValAxis.TIME,
        val.EValAxis.ALEATORY,
    )
    sim_axes = exp_axes
    # Stack two whole parameter scenarios before the aleatory axis. These
    # remain separate epistemic alternatives, each containing its own repeats.
    sim_values = np.stack((sim_values, exp_values), axis=-2)
    sim_axes = (*exp_axes[:-1], val.EValAxis.EPISTEMIC_TRANS, val.EValAxis.ALEATORY)
    # Calibration bounds shift every repeat coherently. They are magnitudes
    # in observable units and describe a shared measurement calibration uncertainty.
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

sim_val_data = val.ValData(sim_arrays)
exp_val_data = val.ValData(exp_arrays)
val_case = val.ValCase(domain, sim_val_data, exp_val_data)

# %%
# 2. Run validation analysis
# --------------------------
# AVM gives a distance in the observable's units. MAVM additionally separates
# positive and negative support mismatch; INCLUDE retains repeated observations.
mavm = val.MetricMAVM(duplicates=val.EMAVMDuplicates.INCLUDE)
result = val.ValidationEngine([val.MetricAVM(), mavm]).run(val_case)

# %%
# 3. Post-processing and inspect data structures
# ----------------------------------------------
# field.axes names each array dimension. The first three axes are always
# (point, component, time), including singleton time for steady comparisons.
# Non-singleton epistemic/offset axes follow in the engine's canonical order.
field = result.get(temperature_tag, val.EMetricType.MAVM)
metric_shape = field.metric.shape
metric_axes = field.axes
if VERBOSE:
    print("Simulation axes:", sim_arrays[temperature_tag].axes)
    print("Simulation shape:", sim_arrays[temperature_tag].data.shape)
    print("Result axes:", metric_axes, "shape:", metric_shape)

# %%
# 4. Analysis and visualisation
# ------------------------------
# Use the SensorSim appearance options so fonts, resolution and line styles
# agree with sensor figures. Each CDF panel overlays simulation and experiment;
# dashed curves show the experimental MAVM confidence bounds.
plot_opts = sens.PlotOptsGeneral(single_fig_scale=0.65)
points = tuple(range(len(coords)))
fig, axes = val.plot_val_cdfs_points(
    val_case,
    result,
    temperature_tag,
    points=points,
    time=4,
    metric=mavm,
    plot_opts=plot_opts,
    labels=val.ValPlotLabels(x_label="Temperature [degC]"),
)
fig.savefig(output_path / "temperature_cdfs.png", bbox_inches="tight")

# %%
# Envelopes describe all retained scenarios at a site. They are pointwise ECDF
# bounds over the conditional distributions. Show every plate sensor.
fig, axes = plt.subplots(4, 3, figsize=(12, 12), layout="constrained")
for point, ax in enumerate(axes.flat):
    val.plot_val_cdf_envelope(
        val_case,
        temperature_tag,
        point=point,
        source=val.EValSource.EXP,
        time=4,
        plot_opts=plot_opts,
        labels=val.ValPlotLabels(title=f"Point {point}: temperature envelope"),
        ax=ax,
    )
fig.savefig(output_path / "temperature_envelopes.png", bbox_inches="tight")

# %%
# Each panel follows one physical sensor through time. Explicit selectors fix
# any remaining epistemic scenario; d+ identifies the directional MAVM quantity.
fig, axes = val.plot_val_metric_time_points(
    result,
    temperature_tag,
    val.EMetricType.MAVM,
    points=points,
    selectors={
        val.EValResAxis.SIM_TRANS: 0,
        val.EValResAxis.EXP_OFFSET: val.EEpisOffset.UPPER,
    },
    part=val.EValMetricPart.D_PLUS,
    plot_opts=plot_opts,
)
fig.savefig(output_path / "temperature_d_plus_time.png", bbox_inches="tight")

# %%
# Overlay validation points on the nominal model in its original xyz geometry.
# The model mesh and sensor positions must share mm coordinates and the plate
# frame. Background nodal scalars (n_nodes,) use their own physical colourbar.
sim_path = dataset.thermomechanical_2d_experiment_paths()[0]
sim_data = mh.ExodusLoader(sim_path).load_all_sim_data()
sens.scale_length_units(1000.0, sim_data, disp_keys=("disp_x", "disp_y"))
background_step = len(sim_data.time) - 1
background_field = val.ValSimField(
    sim_data.node_vars["temperature"][:, background_step],
    background_step,
    sens.SensorDescriptor(name="Temperature", symbol=r"T", units="degC"),
)
scene_opts = sens.VisOptsSimSensors(
    interactive=SHOW_PLOTS,
    camera_position="xy",
    sens_point_size=18,
    sens_label_font_size=14,
    colour_bar_vertical=False,
    colour_bar_font_size=13,
)
point_labels = tuple(sens.SensorDescriptor(tag="TC").create_sensor_tags(len(coords)))
# Blue/red encode negative/positive MAVM directional balance. A white point
# can have nonzero total mismatch because d+ and d- can cancel in this display.
pv_plot = val.plot_val_metric_on_sim(
    result,
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
    sim_coordinate_units="mm",
    sim_frame="plate",
    vis_opts=scene_opts,
    metric_opts=val.ValVisOpts(point_labels=point_labels, labels_always_visible=True),
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
# epis_trans and epis_nuis to inspect a chosen parameter setting; None retains
# every setting. offset=None includes nominal, lower and upper calibration
# shifts. Every returned array is one conditional aleatory sample.
model_quantity = val_case.sim.quantities[temperature_tag]
experiment_quantity = val_case.exp.quantities[temperature_tag]
# Pointwise envelopes choose extrema at each probability and describe the full
# family spread. SUPPORT_SUM selects whole scenarios by the sum of their unique
# support values; lower_index and upper_index identify those original samples.
# Quantile averaging supplies a representative curve through the family.
panel_opts = sens.PlotOptsGeneral(font_leg_size=6)
fig, axes = plt.subplots(4, 3, figsize=(12, 14), layout="constrained")
for point, ax in enumerate(axes.flat):
    val.plot_val_ecdf_family(
        model_quantity,
        point=point,
        component=0,
        time=4,
        plot_opts=panel_opts,
        ax=ax,
        labels=val.ValPlotLabels(
            title=f"Point {point}: conditional model CDFs", x_label="Temperature [degC]"
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
    fig, axes = plt.subplots(4, 3, figsize=(12, 16), layout="constrained")
    for point, ax in enumerate(axes.flat):
        model_scenarios = val.extract_aleatory_scenarios(
            model_quantity,
            point=point,
            component=0,
            time=4,
        )
        experimental_scenarios = val.extract_aleatory_scenarios(
            experiment_quantity,
            point=point,
            component=0,
            time=4,
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
            model_scenarios[model_summary.lower_index],
            model_scenarios[model_summary.upper_index],
            experimental_scenarios[experiment_summary.lower_index],
            experimental_scenarios[experiment_summary.upper_index],
        )
        val.plot_val_mavm_diagnostic(
            model_summary,
            experiment_summary,
            correction=correction,
            view=view,
            plot_opts=panel_opts,
            ax=ax,
            labels=val.ValPlotLabels(
                title=f"Point {point}: MAVM support shifts",
                x_label="Temperature [degC]",
            ),
        )
    fig.savefig(output_path / f"mavm_{view.name.lower()}.png", bbox_inches="tight")
    if not SHOW_PLOTS:
        plt.close(fig)

# %%
# Histograms count physical/scenario cells within each distance bin. Separate
# directional magnitudes expose asymmetry between d+ and d-.
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for part, ax in zip(
    (val.EValMetricPart.D_PLUS, val.EValMetricPart.D_MINUS), axes, strict=True
):
    val.plot_val_metric_distribution(
        result,
        temperature_tag,
        val.EMetricType.MAVM,
        part=part,
        plot_opts=panel_opts,
        ax=ax,
        labels=val.ValPlotLabels(title="Temperature: retained physical/scenario cells"),
    )
fig.savefig(output_path / "directional_metric_distributions.png", bbox_inches="tight")
if not SHOW_PLOTS:
    plt.close(fig)

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
