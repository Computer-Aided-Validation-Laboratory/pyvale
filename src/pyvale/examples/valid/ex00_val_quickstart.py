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
import pyvista as pv
from matplotlib.figure import Figure

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.valid as val
from pyvale.valid import valplot, valvis

matplotlib.use("Agg")

# %%
SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")

output_path = Path.cwd() / "pyvale-output" / "valid_ex0_quickstart"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Load and transform data
# ------------------------------------------------------------
with np.load(dataset.valid_data_dir() / "plate_validation_2d.npz") as saved:
    coords: np.ndarray = saved["coords"].copy()
    times: np.ndarray = saved["times"].copy()
    sim_temp: np.ndarray = saved["sim_temp"].copy()
    exp_temp: np.ndarray = saved["exp_temp"].copy()
    sim_strain: np.ndarray = saved["sim_strain"].copy()
    exp_strain: np.ndarray = saved["exp_strain"].copy()

# %%
temperature_tag = "plate_temperature"
strain_tag = "plate_strain"


observables = {
    temperature_tag: val.ValObservable(
        components=("temperature",),
        geometry=val.EValGeometry.POINT,
        coords=coords,
        times=times,
        phys_units="degC",
        coord_units="mm",
        coord_frame="plate",
        time_units="s",
    ),
    strain_tag: val.ValObservable(
        components=("strain_xx", "strain_yy", "strain_xy"),
        geometry=val.EValGeometry.POINT,
        coords=coords,
        times=times,
        phys_units="1",
        coord_units="mm",
        coord_frame="plate",
        time_units="s",
    ),
}

sim_axes = (
    val.EValAxis.POINT,
    val.EValAxis.COMPONENT,
    val.EValAxis.TIME,
    val.EValAxis.EPISTEMIC_TRANS,
    val.EValAxis.ALEATORY,
)
sim_arrays = {
    temperature_tag: val.ValArray(sim_temp, sim_axes),
    strain_tag: val.ValArray(sim_strain, sim_axes),
}

exp_axes = (
    val.EValAxis.POINT,
    val.EValAxis.COMPONENT,
    val.EValAxis.TIME,
    val.EValAxis.ALEATORY,
)
exp_arrays = {
    temperature_tag: val.ValArray(
        exp_temp,
        exp_axes,
        offsets=val.EpistemicOffsets(lower=0.5, upper=0.75),
    ),
    strain_tag: val.ValArray(
        exp_strain,
        exp_axes,
        offsets=val.EpistemicOffsets(lower=4.0e-6, upper=6.0e-6),
    ),
}

val_case = val.ValCase(observables, sim_arrays, exp_arrays)

# %%
# 2. Run validation analysis
# ------------------------------------------------------------
val_result: val.ValResult = val.ValidationEngine(
    [val.MetricAVM(), val.MetricMAVM()]
).run(val_case)

# %%
# 3. Post-processing and inspect data structures
# ------------------------------------------------------------
metric_field: val.ValMetricField = val_result.get(temperature_tag, val.EMetricType.MAVM)
metric_shape = metric_field.outputs[val.EMetricOutput.D_PLUS].shape
metric_axes = metric_field.axes

print("Simulation axes:", sim_arrays[temperature_tag].axes)
print("Simulation shape:", sim_arrays[temperature_tag].data.shape)
print("Result axes:", metric_axes, "shape:", metric_shape)

# %%
# 4. Analysis and visualisation
# ------------------------------------------------------------

point_idxs = tuple(range(len(coords)))
fig: Figure
axes: np.ndarray
fig, axes = valplot.plot_val_cdfs_points(
    val_case,
    val_result,
    temperature_tag,
    point_idxs=point_idxs,
    time_idx=4,
    metric=val.MetricMAVM(),
    labels=valplot.ValPlotLabels(x_label="Temperature [degC]"),
)
fig.savefig(output_path / "temperature_cdfs.png", bbox_inches="tight")

# %%
fig, axes = valplot.plot_val_cdf_envelope_points(
    val_case,
    temperature_tag,
    source=val.EValSource.EXP,
    time_idx=4,
    labels=valplot.ValPlotLabels(x_label="Temperature [degC]"),
)
fig.savefig(output_path / "temperature_envelopes.png", bbox_inches="tight")

# %%
fig, axes = valplot.plot_val_metric_time_points(
    val_result,
    temperature_tag,
    val.EMetricType.MAVM,
    point_idxs=point_idxs,
    selectors={
        val.EValResAxis.SIM_TRANS: 0,
        val.EValResAxis.EXP_OFFSET: val.EEpisOffset.UPPER,
    },
    output=val.EMetricOutput.D_PLUS,
)
fig.savefig(output_path / "temperature_d_plus_time.png", bbox_inches="tight")

# %%
sim_path = dataset.thermomechanical_2d_experiment_paths()[0]
sim_data: io.SimData = mh.ExodusLoader(sim_path).load_all_sim_data()
sens.scale_length_units(1000.0, sim_data, disp_keys=("disp_x", "disp_y"))

background_time_idx = len(sim_data.time) - 1


pv_plot: pv.Plotter = valvis.plot_val_metric_on_sim(
    val_result,
    temperature_tag,
    val.EMetricType.MAVM,
    selectors={
        val.EValResAxis.SIM_TRANS: 0,
        val.EValResAxis.TIME: 4,
        val.EValResAxis.EXP_OFFSET: val.EEpisOffset.UPPER,
    },
    output=val.EMetricOutput.D_MINUS,
    sim_data=sim_data,
    sim_field="temperature",
    sim_time_idx=background_time_idx,
    descriptor=sens.SensorDescriptor(
        name="Temperature", symbol=r"T", units="degC", tag="TC"
    ),
    sim_coord_units="mm",
    sim_coord_frame="plate",
    vis_opts=valvis.create_val_scene_opts(interactive=SHOW_PLOTS),
    metric_opts=valvis.ValMetricVisOpts(labels_always_visible=True),
)
pv_plot.screenshot(output_path / "temperature_mavm_d_minus_on_sim.png")


# %%
if SHOW_PLOTS:
    pv_plot.show()
    plt.show()
else:
    plt.close("all")

pv_plot.close()
