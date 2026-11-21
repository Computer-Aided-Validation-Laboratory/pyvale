# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Transient plate validation
================================================================================

Introduce physical time as a separate comparison axis. The same 12 sensors
observe temperature and strain at five times, with six repeats at each time.
Learn to follow metric histories and select a matching simulation snapshot.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyvale.data as dataset
import pyvale.validation as vl


# %%
# Enable interactive windows with SHOW_PLOTS.
matplotlib.use("Agg")
SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")

output_path = Path.cwd() / "pyvale-output" / "valid_ex3_transient"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Load and transform data
# -----------------------------------------
# The plate has 12 sensor locations on a 4 by 3 grid: x=10..90 mm,
# y=10..40 mm, z=0. Each site has a thermocouple (one temperature component)
# and an in-plane strain rosette (xx, yy, xy). Six virtual repeat experiments
# supply aleatory samples at five common physical times.
# Exp. is constructed below from the same aleatory grid plus a known bias.
# This isolates systematic model discrepancy from Monte Carlo sampling noise.
# Prepared arrays already follow the validation axis order declared below.
# Temperature has one component; strain has three components (xx, yy, xy).
with np.load(dataset.valid_data_dir() / "plate_validation_2d_nominal.npz") as saved:
    coords: np.ndarray = saved["coords"].copy()
    times: np.ndarray = saved["times"].copy()
    sim_temp: np.ndarray = saved["sim_temp"].copy()  # (12, 1, 5, 6)
    sim_strain: np.ndarray = saved["sim_strain"].copy()

# Controlled synthetic discrepancy: zero initially, growing towards the right
# edge. At the final time the left sensors differ by 2 degC and the right by
# 5 degC. These are new in-memory arrays; no packaged data are changed.
x_fraction = (coords[:, 0] - np.min(coords[:, 0])) / np.ptp(coords[:, 0])
time_fraction = (times - times[0]) / (times[-1] - times[0])
temperature_bias = (2.0 + 3.0 * x_fraction[:, None]) * time_fraction
exp_temp = sim_temp + temperature_bias[:, None, :, None]
strain_bias = (20.0e-6 * x_fraction[:, None] + 10.0e-6) * time_fraction
exp_strain = sim_strain + strain_bias[:, None, :, None]

# %%
# A shared observables registers units, component order and physical coordinates.
# Validation compares corresponding physical cells, retaining distinct point,
# component, time and repeat-experiment axes.

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
        phys_units="1",
        coord_units="mm",
        time_units="s",
    ),
}

# Physical axes identify each comparison cell; ALEATORY holds its repeated
# observations. The declared axis order matches the arrays in the archive.

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
# 2. Run validation analysis
# --------------------------
# AVM gives a distance in the observable's units. MAVM additionally separates
# positive and negative support mismatch; INCLUDE retains repeated observations.
valid_result: vl.ValidResult = vl.ValidEngine(
    [vl.MetricAVM(), vl.MetricMAVM()]
).run(valid_case)

# %%
# 3. Post-processing and inspect data structures
# ----------------------------------------------
# metric_field.axes names each array dimension. The first three axes are always
# (point, component, time), including singleton time for steady comparisons.
# Non-singleton epistemic/offset axes follow in the engine's canonical order.
metric_field: vl.MetricField = valid_result.metrics[temperature_tag][
    vl.EMetricType.MAVM
]
metric_shape = metric_field.outputs[vl.EMetricOut.D_PLUS].shape
metric_axes = metric_field.axes

print(80*"-")
print("Simulation axes:", sim_arrays[temperature_tag].axes)
print("Simulation shape:", sim_arrays[temperature_tag].data.shape)
print("Result axes:", metric_axes, "shape:", metric_shape)
print(80*"-")

# %%
# 4. Analysis and visualisation
# ------------------------------
# Use the SensorSim appearance options so fonts, resolution and line styles
# agree with sensor figures. Each panel compares the full selected sim and exp
# epistemic envelopes, or one ECDF when only one scenario is available.
point_idxs = tuple(range(len(coords)))
ecdf_plot: vl.ECDFPlot = vl.plot_sim_exp_cdfs(
    valid_case,
    temperature_tag,
    point_idxs=point_idxs,
    time_idx=4,
    valid_plot_opts=vl.ValidPlotOpts(x_label="Temperature [degC]"),
)
ecdf_plot.figure.savefig(output_path / "temperature_cdfs.png", bbox_inches="tight")

# AVM histories recover the imposed linear drift directly; compare these with
# both MAVM directions, which additionally include confidence shifts.
fig, _ = vl.plot_metric_time_points(
    valid_result, temperature_tag, vl.EMetricType.AVM, point_idxs=point_idxs
)
fig.savefig(output_path / "temperature_avm_time.png", bbox_inches="tight")

# %%
# Each panel follows one physical sensor through time. Explicit selectors fix
# any remaining epistemic scenario.
fig, _ = vl.plot_metric_time_points(
    valid_result,
    temperature_tag,
    vl.EMetricType.MAVM,
    point_idxs=point_idxs,
)
fig.savefig(output_path / "temperature_mavm_time.png", bbox_inches="tight")

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
