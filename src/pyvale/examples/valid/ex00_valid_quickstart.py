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
from pyvale import valid
from pyvale.valid import validplot, validvis

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
    temperature_tag: valid.ValidObservable(
        components=("temperature",),
        geometry=valid.EValidGeometry.POINT,
        coords=coords,
        times=times,
        phys_units="degC",
        coord_units="mm",
        time_units="s",
    ),
    strain_tag: valid.ValidObservable(
        components=("strain_xx", "strain_yy", "strain_xy"),
        geometry=valid.EValidGeometry.POINT,
        coords=coords,
        times=times,
        phys_units="1",
        coord_units="mm",
        time_units="s",
    ),
}

sim_axes = (
    valid.EValidAxis.POINT,
    valid.EValidAxis.COMPONENT,
    valid.EValidAxis.TIME,
    valid.EValidAxis.EPISTEMIC_TRANS,
    valid.EValidAxis.ALEATORY,
)
sim_arrays = {
    temperature_tag: valid.ValidArray(sim_temp, sim_axes),
    strain_tag: valid.ValidArray(sim_strain, sim_axes),
}

exp_axes = (
    valid.EValidAxis.POINT,
    valid.EValidAxis.COMPONENT,
    valid.EValidAxis.TIME,
    valid.EValidAxis.ALEATORY,
)
exp_arrays = {
    temperature_tag: valid.ValidArray(
        exp_temp,
        exp_axes,
        offsets=valid.EpistemicOffsets(lower=0.5, upper=0.75),
    ),
    strain_tag: valid.ValidArray(
        exp_strain,
        exp_axes,
        offsets=valid.EpistemicOffsets(lower=4.0e-6, upper=6.0e-6),
    ),
}

valid_case = valid.ValidCase(observables, sim_arrays, exp_arrays)

# %%
# 2. Run validation analysis
# ------------------------------------------------------------
valid_result: valid.ValidResult = valid.ValidEngine(
    [valid.MetricAVM(), valid.MetricMAVM()]
).run(valid_case)

# %%
# 3. Post-processing and inspect data structures
# ------------------------------------------------------------
metric_field: valid.ValidMetricField = valid_result.metrics[temperature_tag][
    valid.EMetricType.MAVM
]
metric_shape = metric_field.outputs[valid.EMetricOutput.D_PLUS].shape
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
ecdf_plot: validplot.ValidECDFPlot = validplot.plot_sim_exp_cdfs(
    valid_case,
    temperature_tag,
    point_idxs=point_idxs,
    time_idx=4,
    valid_plot_opts=validplot.ValidPlotOpts(x_label="Temperature [degC]"),
)
fig, axes = ecdf_plot.figure, ecdf_plot.axes
fig.savefig(output_path / "temperature_cdfs.png", bbox_inches="tight")

# %%
ecdf_plot: validplot.ValidECDFPlot = validplot.plot_exp_ecdfs(
    valid_case,
    temperature_tag,
    time_idx=4,
    valid_plot_opts=validplot.ValidPlotOpts(x_label="Temperature [degC]"),
    point_idxs=None,
    offset=None,
)
fig, axes = ecdf_plot.figure, ecdf_plot.axes
fig.savefig(output_path / "temperature_envelopes.png", bbox_inches="tight")

# %%
for metric_output, direction in (
    (valid.EMetricOutput.D_PLUS, "d_plus"),
    (valid.EMetricOutput.D_MINUS, "d_minus"),
):
    fig, axes = validplot.plot_valid_metric_time_points(
        valid_result,
        temperature_tag,
        valid.EMetricType.MAVM,
        point_idxs=point_idxs,
        selectors={
            valid.EValidResAxis.SIM_TRANS: 0,
            valid.EValidResAxis.EXP_OFFSET: valid.EEpisOffset.UPPER,
        },
        output=metric_output,
    )
    fig.savefig(output_path / f"temperature_{direction}_time.png", bbox_inches="tight")

# %%
sim_path = dataset.thermomechanical_2d_experiment_paths()[0]
sim_data: io.SimData = mh.ExodusLoader(sim_path).load_all_sim_data()
sens.scale_length_units(1000.0, sim_data, disp_keys=("disp_x", "disp_y"))

assert sim_data.time is not None
background_time_idx = sim_data.time.size - 1


for metric_output, direction in (
    (valid.EMetricOutput.D_PLUS, "d_plus"),
    (valid.EMetricOutput.D_MINUS, "d_minus"),
):
    pv_plot: pv.Plotter = validvis.plot_valid_metric_on_sim(
        valid_result,
        temperature_tag,
        valid.EMetricType.MAVM,
        selectors={
            valid.EValidResAxis.SIM_TRANS: 0,
            valid.EValidResAxis.TIME: 4,
            valid.EValidResAxis.EXP_OFFSET: valid.EEpisOffset.UPPER,
        },
        output=metric_output,
        sim_data=sim_data,
        sim_field="temperature",
        sim_time_idx=background_time_idx,
        descriptor=sens.SensorDescriptor(
            name="Temperature", symbol=r"T", units="degC", tag="TC"
        ),
        sim_coord_units="mm",
        vis_opts=validvis.create_valid_scene_opts(interactive=SHOW_PLOTS),
        metric_opts=validvis.ValidMetricVisOpts(labels_always_visible=True),
    )
    pv_plot.screenshot(output_path / f"temperature_mavm_{direction}_on_sim.png")

    # %%
    if SHOW_PLOTS:
        pv_plot.show()
        plt.show()
    else:
        plt.close("all")

    pv_plot.close()
