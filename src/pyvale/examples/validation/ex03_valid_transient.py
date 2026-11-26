# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Transient validation
================================================================================

Follow a plate discrepancy over time using ECDF snapshots and metric
histories.
"""

from itertools import count
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyvale.data as dataset
import pyvale.validation as vl


# %%
matplotlib.use("Agg")

SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")

output_path = Path.cwd() / "pyvale-output" / "valid_ex03_transient"
output_path.mkdir(parents=True, exist_ok=True)
fig_nums = count(1)

# %%
# 1. Prepare the transient samples
# --------------------------------
#
# The arrays are ordered by point, component, time, and aleatory repeat. The
# synthetic temperature bias grows from zero to 2–5 degC, depending on the
# sensor position. The two-dimensional bias has point and time axes and
# broadcasts across components and repeats.
with np.load(dataset.valid_data_dir() / "plate_validation_2d_nominal.npz") as saved:
    coords: np.ndarray = saved["coords"].copy()
    times: np.ndarray = saved["times"].copy()
    sim_temp: np.ndarray = saved["sim_temp"].copy()  # (12, 1, 5, 6)
    sim_strain: np.ndarray = saved["sim_strain"] * 1.0e3

# Increase the synthetic repeat scatter while preserving the mean strain.
strain_repeat_spread_factor = 3.0
sim_strain_mean = sim_strain.mean(axis=-1, keepdims=True)
sim_strain = sim_strain_mean + strain_repeat_spread_factor * (
    sim_strain - sim_strain_mean
)

x_fraction = (coords[:, 0] - np.min(coords[:, 0])) / np.ptp(coords[:, 0])
time_fraction = (times - times[0]) / (times[-1] - times[0])
temperature_bias = (2.0 + 3.0 * x_fraction[:, None]) * time_fraction
exp_temp = sim_temp + temperature_bias[:, None, :, None]
strain_bias = (0.020 * x_fraction[:, None] + 0.010) * time_fraction
exp_strain = sim_strain + strain_bias[:, None, :, None]

# %%
# 2. Pair data on a shared time grid
# ----------------------------------
#
# Use the observable to record physical sample times and sensor coordinates.
# Align simulation and experiment to this grid before creating the case.
# Temperature uses degC and strain uses millistrain.
temperature_tag = "plate_temperature"
strain_tag = "plate_strain"

observables = {
    temperature_tag: vl.ValidObservable(
        components=("temperature",),
        geometry=vl.EValidGeometry.POINT,
        coords=coords,
        times=times,
        phys_units="degC",
        coord_units="mm",
        time_units="s",
    ),
    strain_tag: vl.ValidObservable(
        components=("strain_xx", "strain_yy", "strain_xy"),
        geometry=vl.EValidGeometry.POINT,
        coords=coords,
        times=times,
        phys_units="mε",
        coord_units="mm",
        time_units="s",
    ),
}

sim_axes = (
    vl.EValidAxis.POINT,
    vl.EValidAxis.COMPONENT,
    vl.EValidAxis.TIME,
    vl.EValidAxis.ALEATORY,
)

exp_axes = (
    vl.EValidAxis.POINT,
    vl.EValidAxis.COMPONENT,
    vl.EValidAxis.TIME,
    vl.EValidAxis.ALEATORY,
)

sim_arrays = {
    temperature_tag: vl.ValidArray(sim_temp, sim_axes),
    strain_tag: vl.ValidArray(sim_strain, sim_axes),
}
exp_arrays = {
    temperature_tag: vl.ValidArray(exp_temp, exp_axes),
    strain_tag: vl.ValidArray(exp_strain, exp_axes),
}

valid_case = vl.ValidCase(observables, sim_arrays, exp_arrays)

# %%
# 3. Calculate metrics over time
# ------------------------------
#
# Each metric calculation compares the aleatory samples at one point,
# component, and time. The time axis remains in the metric field so you can
# plot a history directly.
valid_result: vl.ValidResult = vl.ValidEngine(
    [vl.MetricAVM(), vl.MetricMAVM()]
).run(valid_case)

# %%
metric_field: vl.MetricField = valid_result.metrics[temperature_tag][
    vl.EMetricType.MAVM
]
metric_shape = metric_field.metric_vals[vl.EMetricOut.D_PLUS].shape
metric_axes = metric_field.axes

print(80*"-")
print("Simulation axes:", sim_arrays[temperature_tag].axes)
print("Simulation shape:", sim_arrays[temperature_tag].data.shape)
print("Result axes:", metric_axes, "shape:", metric_shape)
print(80*"-")

# %%
# 4. Plot snapshots and histories
# -------------------------------
#
# Select one time index for the ECDF comparison. Plot AVM over physical time
# to track the growing bias. The MAVM history shows both directional outputs
# on each point panel.
point_idxs = tuple(range(len(coords)))
ecdf_plot: vl.ECDFPlot = vl.plot_sim_exp_cdfs(
    valid_case,
    temperature_tag,
    point_idxs=point_idxs,
    time_idx=4,
    valid_plot_opts=vl.ValidPlotOpts(x_label="Temperature [degC]"),
)
fig_file = f"val_ex03_fig{next(fig_nums):02d}_temperature_cdfs.png"
ecdf_plot.figure.savefig(output_path / fig_file, bbox_inches="tight")

fig, _ = vl.plot_metric_time_points(
    valid_result, temperature_tag, vl.EMetricType.AVM, point_idxs=point_idxs
)
fig_file = f"val_ex03_fig{next(fig_nums):02d}_temperature_avm_time.png"
fig.savefig(output_path / fig_file, bbox_inches="tight")

# %%
fig, _ = vl.plot_metric_time_points(
    valid_result,
    temperature_tag,
    vl.EMetricType.MAVM,
    point_idxs=point_idxs,
)
fig_file = f"val_ex03_fig{next(fig_nums):02d}_temperature_mavm_time.png"
fig.savefig(output_path / fig_file, bbox_inches="tight")

# %%
# .. image:: /_static/val_ex03_fig01_temperature_cdfs.png
#    :alt: Transient plate temperature ECDFs at the selected time
#    :width: 100%
#    :align: center
#
# .. image:: /_static/val_ex03_fig03_temperature_mavm_time.png
#    :alt: Directional temperature MAVM over time at each sensor
#    :width: 100%
#    :align: center

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
