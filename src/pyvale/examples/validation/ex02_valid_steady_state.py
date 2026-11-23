# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Steady state validation
================================================================================

Validate steady-state plate temperature and strain and map the discrepancy to
the sensor locations.
"""

from itertools import count
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.validation as vl

# %%
matplotlib.use("Agg")

SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")

output_path = Path.cwd() / "pyvale-output" / "valid_ex02_steady_state"
output_path.mkdir(parents=True, exist_ok=True)
fig_nums = count(1)

# %%
# 1. Prepare the steady-state samples
# -----------------------------------
#
# The sample arrays are ordered by point, component, and aleatory repeat. Each
# row of ``coords`` supplies the x, y, and z location of one point. The
# synthetic experiment adds a bias that grows along x, so the temperature
# discrepancy has a known spatial pattern. Strain is expressed in millistrain.
with np.load(
    dataset.valid_data_dir() / "plate_validation_2d_steady_nominal.npz"
) as saved:
    coords: np.ndarray = saved["coords"].copy()
    sim_temp: np.ndarray = saved["sim_temp"].copy()  # (12, 1, 6)
    sim_strain: np.ndarray = saved["sim_strain"] * 1.0e3  # (12, 3, 6)

# Increase the synthetic repeat scatter while preserving the mean strain.
strain_repeat_spread_factor = 3.0
sim_strain_mean = sim_strain.mean(axis=-1, keepdims=True)
sim_strain = sim_strain_mean + strain_repeat_spread_factor * (
    sim_strain - sim_strain_mean
)

x_fraction = (coords[:, 0] - np.min(coords[:, 0])) / np.ptp(coords[:, 0])
temperature_bias = 3.0 * x_fraction
exp_temp = sim_temp + temperature_bias[:, None, None]
exp_strain = (
    sim_strain
    + x_fraction[:, None, None] * np.array([0.020, -0.010, 0.005])[None, :, None]
)

# %%
# 2. Register the sensor locations
# --------------------------------
#
# Use ``POINT`` observables to associate each sample row with its sensor
# coordinates. Temperature has one component and strain has three. The same
# observable tags connect both sources to their coordinate and unit metadata.
temperature_tag = "plate_temperature"
strain_tag = "plate_strain"

observables = {
    temperature_tag: vl.ValidObservable(
        components=("temperature",),
        geometry=vl.EValidGeometry.POINT,
        coords=coords,
        phys_units="degC",
        coord_units="mm",
    ),
    strain_tag: vl.ValidObservable(
        components=("strain_xx", "strain_yy", "strain_xy"),
        geometry=vl.EValidGeometry.POINT,
        coords=coords,
        phys_units="mε",
        coord_units="mm",
    ),
}

# %%
sim_axes = (
    vl.EValidAxis.POINT,
    vl.EValidAxis.COMPONENT,
    vl.EValidAxis.ALEATORY,
)
exp_axes = (
    vl.EValidAxis.POINT,
    vl.EValidAxis.COMPONENT,
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
# 3. Calculate validation metrics
# -------------------------------
#
# Run AVM and MAVM at each point and component. Metric fields include a
# singleton time dimension for this steady state. Inspect the named axes and
# select a directional MAVM output to read its values.
valid_result: vl.ValidResult = vl.ValidEngine(
    [vl.MetricAVM(), vl.MetricMAVM(alpha=0.5)]
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
# 4. Compare the ECDFs
# --------------------
#
# Select the sensor indices to compare the distributions side by side. Each
# panel corresponds to one point for the selected component.
point_idxs = tuple(range(len(coords)))

ecdf_plot: vl.ECDFPlot = vl.plot_sim_exp_cdfs(
    valid_case,
    temperature_tag,
    point_idxs=point_idxs,
    valid_plot_opts=vl.ValidPlotOpts(x_label="Temperature [degC]"),
)
fig_file = f"val_ex02_fig{next(fig_nums):02d}_temperature_cdfs.png"
ecdf_plot.figure.savefig(output_path / fig_file, bbox_inches="tight")

# %%
# .. image:: /_static/val_ex02_fig01_temperature_cdfs.png
#    :alt: Steady-state temperature ECDFs across the plate sensors
#    :width: 100%
#    :align: center

# %%
# 5. Map MAVM onto the plate
# --------------------------
#
# Use the temperature field as a spatial backdrop for the metric markers. The
# selected metric output and the mesh coordinates share the same physical
# frame and length units.
sim_path = dataset.thermomechanical_2d_experiment_paths()[0]
sim_data: io.SimData = mh.ExodusLoader(sim_path).load_all_sim_data()
sens.scale_length_units(1000.0, sim_data, disp_keys=("disp_x", "disp_y"))

background_time_idx = sim_data.time.size - 1

pv_plot: pv.Plotter = vl.plot_metric_on_sim(
    valid_result,
    temperature_tag,
    vl.EMetricType.AVM,
    sim_data=sim_data,
    sim_field="temperature",
    sim_time_idx=background_time_idx,
    descriptor=sens.SensorDescriptor(
        name="Temperature", symbol=r"T", units="degC", tag="TC"
    ),
    sim_coord_units="mm",
    vis_opts=vl.create_valid_scene_opts(interactive=SHOW_PLOTS),
    metric_opts=vl.MetricVisOpts(labels_always_visible=True),
)

pv_plot.screenshot(
    output_path / f"val_ex02_fig{next(fig_nums):02d}_temperature_avm_on_sim.png",
)

if SHOW_PLOTS:
    pv_plot.show()
pv_plot.close()


# %%
# .. image:: /_static/val_ex02_fig02_temperature_avm_on_sim.png
#    :alt: AVM discrepancy at the plate sensor locations
#    :width: 100%
#    :align: center

# %%
# 6. Show the MAVM support span
# -----------------------------
#
# Find the shifted simulation support from the calculated result and overlay
# it on the ECDF comparison.
supports = vl.find_metric_epis_supports(
    valid_case, valid_result, temperature_tag, vl.EMetricType.MAVM
)
ecdf_plot = vl.plot_sim_exp_cdfs(valid_case, temperature_tag)
vl.add_support_overlay(ecdf_plot, supports)
fig_file = f"val_ex02_fig{next(fig_nums):02d}_mavm_support_envelope.png"
ecdf_plot.figure.savefig(output_path / fig_file, bbox_inches="tight")
if not SHOW_PLOTS:
    plt.close(ecdf_plot.figure)


if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
