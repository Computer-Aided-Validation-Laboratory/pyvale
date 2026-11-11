# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Validation quickstart
================================================================================

Transient temperature and strain validation with aleatory repeats, epistemic
model scenarios, calibration offsets, MAVM diagnostics and simulation overlays.
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
SHOW_PLOTS = False
VERBOSE = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")
output_path = Path.cwd() / "pyvale-output" / "valid_ex0_quickstart"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Load and transform data (pre-processing)
# ------------------------------------------------------------
with np.load(dataset.valid_data_dir() / "plate_validation_2d.npz") as saved:
    coords = saved["coords"].copy()
    times = saved["times"].copy()
    nominal_temp = saved["nominal_temperature"].copy()
    measured_temp = saved["perturbed_temperature"].copy()
    nominal_strain = saved["nominal_strain"].copy()
    measured_strain = saved["perturbed_strain"].copy()

# %%
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
    sim_values = np.moveaxis(nominal, 0, -1)
    exp_values = np.moveaxis(measured, 0, -1)
    exp_axes = (
        val.EValAxis.POINT,
        val.EValAxis.COMPONENT,
        val.EValAxis.TIME,
        val.EValAxis.ALEATORY,
    )
    sim_axes = exp_axes
    sim_values = np.stack((sim_values, exp_values), axis=-2)
    sim_axes = (*exp_axes[:-1], val.EValAxis.EPISTEMIC_TRANS, val.EValAxis.ALEATORY)
    magnitude = 0.5 if observable_tag == temperature_tag else 4.0e-6
    if observable_tag == temperature_tag:
        spatial = coords[:, 0, None] / 100.0
        temporal = times[None, :] / times[-1]
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
# ------------------------------------------------------------
mavm = val.MetricMAVM(duplicates=val.EMAVMDuplicates.INCLUDE)
result = val.ValidationEngine([val.MetricAVM(), mavm]).run(val_case)

# %%
# 3. Post-processing and inspect data structures
# ------------------------------------------------------------
field = result.get(temperature_tag, val.EMetricType.MAVM)
metric_shape = field.metric.shape
metric_axes = field.axes
if VERBOSE:
    print("Simulation axes:", sim_arrays[temperature_tag].axes)
    print("Simulation shape:", sim_arrays[temperature_tag].data.shape)
    print("Result axes:", metric_axes, "shape:", metric_shape)

# %%
# 4. Analysis and visualisation
# ------------------------------------------------------------
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
if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
