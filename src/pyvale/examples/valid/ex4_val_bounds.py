# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Coherent epistemic measurement offsets
================================================================================

Follow preprocessing, validation, post-processing and visualisation explicitly.
The measurements are synthetic SensorSim experiments, not laboratory data.
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
output_path = Path.cwd() / "pyvale-output" / "valid_ex4_bounds"
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
# Validation compares corresponding cells; it never flattens points or times
# into the repeat-experiment axis.
geometry = val.ValGeometry(
    val.EValGeometry.POINT, coords, coordinate_units="mm", frame="plate"
)
domain = val.ValDomain(
    {
        "temperature": val.ValObservable(
            geometry,
            val.EValOperator.SAMPLE,
            ("temperature",),
            times,
            "degC",
            time_units="s",
        ),
        "strain": val.ValObservable(
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
for observable, nominal, measured in (
    ("temperature", nominal_temp, measured_temp),
    ("strain", nominal_strain, measured_strain),
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
    # remain separate epistemic alternatives, not extra repeat experiments.
    sim_values = np.stack((sim_values, exp_values), axis=-2)
    sim_axes = (*exp_axes[:-1], val.EValAxis.EPISTEMIC_TRANS, val.EValAxis.ALEATORY)
    # Calibration bounds shift every repeat coherently. They are magnitudes
    # in observable units, not confidence intervals or independent random noise.
    magnitude = 0.5 if observable == "temperature" else 4.0e-6
    if observable == "temperature":
        spatial = coords[:, 0, None] / 100.0  # (point, singleton time)
        temporal = times[None, :] / times[-1]  # (singleton point, time)
        bound = val.EpistemicOffsets(
            lower=magnitude * (1.0 + 0.2 * spatial + 0.2 * temporal),
            upper=1.5 * magnitude * (1.0 + 0.1 * spatial + 0.3 * temporal),
            axes=(val.EValAxis.POINT, val.EValAxis.TIME),
        )
    else:
        bound = val.EpistemicOffsets(lower=magnitude, upper=1.5 * magnitude)
    sim_arrays[observable] = val.ValArray(sim_values, sim_axes)
    exp_arrays[observable] = val.ValArray(exp_values, exp_axes, offsets=bound)

val_case = val.ValCase(domain, val.ValData(sim_arrays), val.ValData(exp_arrays))

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
field = result.get("temperature", val.EMetricType.MAVM)
metric_shape = field.metric.shape
metric_axes = field.axes
if VERBOSE:
    print("Simulation axes:", sim_arrays["temperature"].axes)
    print("Simulation shape:", sim_arrays["temperature"].data.shape)
    print("Result axes:", metric_axes, "shape:", metric_shape)

# %%
# 4. Analysis and visualisation
# ------------------------------
# Use the SensorSim appearance options so fonts, resolution and line styles
# agree with sensor figures. Each CDF panel overlays simulation and experiment;
# dashed curves are the experimental MAVM confidence bounds, not model bounds.
plot_opts = sens.PlotOptsGeneral(single_fig_scale=0.65)
points = tuple(range(len(coords)))
fig, axes = val.plot_val_cdfs_points(
    val_case,
    result,
    "temperature",
    points=points,
    time=4,
    metric=mavm,
    plot_opts=plot_opts,
    labels=val.ValPlotLabels(x_label="Temperature [degC]"),
)
fig.savefig(output_path / "temperature_cdfs.png", bbox_inches="tight")

# %%
# Envelopes describe all retained scenarios at a site. They are pointwise ECDF
# bounds, not a new pooled probability distribution. Show every plate sensor.
fig, axes = plt.subplots(4, 3, figsize=(12, 12), layout="constrained")
for point, ax in enumerate(axes.flat):
    val.plot_val_cdf_envelope(
        val_case,
        "temperature",
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
    "temperature",
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
    "temperature",
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
model_quantity = val_case.sim.quantities["temperature"]
experiment_quantity = val_case.exp.quantities["temperature"]
panel_opts = sens.PlotOptsGeneral(font_leg_size=6)
diagnostic_views = (
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

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
