# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Steady plate validation
================================================================================

Extend ex1 to 12 physical locations and two observables: temperature and
three component strain in 2D. Learn how shared geometry and observable tags organise
different measurements in one ValidCase. The observations come from SensorSim.
"""

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

matplotlib.use("Agg")

# %%
# Console inspection is printed below. Enable interactive windows with SHOW_PLOTS.
SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")

output_path = Path.cwd() / "pyvale-output" / "valid_ex2_steady"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Load and transform data
# -----------------------------------------
# The plate has 12 sensor locations on a 4 by 3 grid: x=10..90 mm,
# y=10..40 mm, z=0. Each site has a thermocouple (one temperature component)
# and an in-plane strain rosette (xx, yy, xy). Six virtual repeat experiments
# supply aleatory samples at the final simulation state.
# Exp. is constructed from the sim aleatory grid plus a known spatial bias.
# The random-looking discrepancy in the archive is deliberately not used.
# Prepared arrays already follow the validation axis order declared below.
# Temperature has one component; strain has three components (xx, yy, xy).

with np.load(
    dataset.valid_data_dir() / "plate_validation_2d_steady_nominal.npz"
) as saved:
    coords: np.ndarray = saved["coords"].copy()
    sim_temp: np.ndarray = saved["sim_temp"].copy()  # (12, 1, 6)
    sim_strain: np.ndarray = saved["sim_strain"].copy()  # (12, 3, 6)

# Add 0..3 degC from the left sensor edge to the right. Reusing the sim sample
# grid makes the AVM exactly this bias, so the spatial map has an interpretable
# gradient. The strain components have a smaller, signed synthetic edge bias.
x_fraction = (coords[:, 0] - np.min(coords[:, 0])) / np.ptp(coords[:, 0])
temperature_bias = 3.0 * x_fraction
exp_temp = sim_temp + temperature_bias[:, None, None]
exp_strain = (
    sim_strain
    + x_fraction[:, None, None] * np.array([20.0e-6, -10.0e-6, 5.0e-6])[None, :, None]
)

# %%
# A shared observables dictionary registers units, component order and physical
# coordinates. Validation compares corresponding physical cells, retaining distinct
# point, component, time and repeat experiment axes.
# Each ValidObservable stores its geometry metadata directly. The observables share
# the same coordinate array, shaped (12 points, 3), without copying it. Each supplies
# its own component tuple and units. We choose distinct tags below; the same tags key both
# source dictionaries and the result fields.

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
        phys_units="1",
        coord_units="mm",
    ),
}

# %%
# Physical axes identify each comparison cell; ALEATORY holds its repeated
# observations. The declared axis order matches the arrays in the archive.

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
# 2. Run validation analysis
# --------------------------
# AVM gives a distance in the observable's units. MAVM additionally separates
# positive and negative support mismatch. Alpha=0.5 gives a zero Student’s t
# shift so this example isolates the prescribed bias between the empirical CDFs.

valid_result: vl.ValidResult = vl.ValidEngine(
    [vl.MetricAVM(), vl.MetricMAVM(alpha=0.5)]
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
    valid_plot_opts=vl.ValidPlotOpts(x_label="Temperature [degC]"),
)
ecdf_plot.figure.savefig(output_path / "temperature_cdfs.png", bbox_inches="tight")

# %%
# Compare spatial mismatch at the same final physical time. AVM retains degC
# units; a small value means the two empirical temperature distributions agree.
#
# Overlay validation points on the nominal model in its original xyz geometry.
# The model mesh and sensor positions use the same coordinates in mm.
# Background nodal scalars (n_nodes,) have a physical colourbar.

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

pv_plot.screenshot(output_path / "temperature_avm_on_sim.png")

if SHOW_PLOTS:
    pv_plot.show()
pv_plot.close()


# %%
# Figures are saved even when interactive plotting is disabled. Keeping the
# complete analysis in its own directory prevents clashes with other examples.
# Show the MAVM support envelope across both directions.
supports = vl.find_metric_epis_supports(
    valid_case, valid_result, temperature_tag, vl.EMetricType.MAVM
)
ecdf_plot = vl.plot_sim_exp_cdfs(valid_case, temperature_tag)
vl.add_support_overlay(ecdf_plot, supports)
ecdf_plot.figure.savefig(
    output_path / "mavm_support_envelope.png", bbox_inches="tight"
)
if not SHOW_PLOTS:
    plt.close(ecdf_plot.figure)

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
