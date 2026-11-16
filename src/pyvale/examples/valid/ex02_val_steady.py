# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Steady plate validation
================================================================================

Extend ex1 to 12 physical locations and two observables: temperature and
three component strain in 2D. Learn how shared geometry and observable tags organise
different measurements in one ValCase. The observations come from SensorSim.
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
import pyvale.valid as val
from pyvale.valid import valplot, valvis

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
# The second simulation has conductivity and thermal expansion reduced by 10%.
# We treat its noisy measurements as the experiment to expose model discrepancy.
# Prepared arrays already follow the validation axis order declared below.
# Temperature has one component; strain has three components (xx, yy, xy).

with np.load(
    dataset.valid_data_dir() / "plate_validation_2d_steady_nominal.npz"
) as saved:
    coords: np.ndarray = saved["coords"].copy()
    sim_temp: np.ndarray = saved["sim_temp"].copy()  # (12, 1, 6)
    exp_temp: np.ndarray = saved["exp_temp"].copy()  # (12, 1, 6)
    sim_strain: np.ndarray = saved["sim_strain"].copy()  # (12, 3, 6)
    exp_strain: np.ndarray = saved["exp_strain"].copy()  # (12, 3, 6)

# %%
# A shared observables dictionary registers units, component order and physical
# coordinates. Validation compares corresponding physical cells, retaining distinct
# point, component, time and repeat experiment axes.
# Each ValObservable stores its geometry metadata directly. The observables share
# the same coordinate array, shaped (12 points, 3), without copying it. Each supplies
# its own component tuple and units. We choose distinct tags below; the same tags key both
# source dictionaries and the result fields.

temperature_tag = "plate_temperature"
strain_tag = "plate_strain"

observables = {
    temperature_tag: val.ValObservable(
        components=("temperature",),
        geometry=val.EValGeometry.POINT,
        coords=coords,
        phys_units="degC",
        coord_units="mm",
    ),
    strain_tag: val.ValObservable(
        components=("strain_xx", "strain_yy", "strain_xy"),
        geometry=val.EValGeometry.POINT,
        coords=coords,
        phys_units="1",
        coord_units="mm",
    ),
}

# %%
# Physical axes identify each comparison cell; ALEATORY holds its repeated
# observations. The declared axis order matches the arrays in the archive.

sim_axes = (
    val.EValAxis.POINT,
    val.EValAxis.COMPONENT,
    val.EValAxis.ALEATORY,
)
exp_axes = (
    val.EValAxis.POINT,
    val.EValAxis.COMPONENT,
    val.EValAxis.ALEATORY,
)

sim_arrays: dict[str, val.ValArray] = {}
exp_arrays: dict[str, val.ValArray] = {}

for observable_tag, sim_values, exp_values in (
    (temperature_tag, sim_temp, exp_temp),
    (strain_tag, sim_strain, exp_strain),
):
    sim_arrays[observable_tag] = val.ValArray(sim_values, sim_axes)
    exp_arrays[observable_tag] = val.ValArray(exp_values, exp_axes)

val_case = val.ValCase(observables, sim_arrays, exp_arrays)

# %%
# 2. Run validation analysis
# --------------------------
# AVM gives a distance in the observable's units. MAVM additionally separates
# positive and negative support mismatch; INCLUDE retains repeated observations.

val_result: val.ValResult = val.ValidationEngine(
    [val.MetricAVM(), val.MetricMAVM()]
).run(val_case)

# %%
# 3. Post-processing and inspect data structures
# ----------------------------------------------
# metric_field.axes names each array dimension. The first three axes are always
# (point, component, time), including singleton time for steady comparisons.
# Non-singleton epistemic/offset axes follow in the engine's canonical order.

metric_field: val.ValMetricField = val_result.metrics[temperature_tag][
    val.EMetricType.MAVM
]
metric_shape = metric_field.outputs[val.EMetricOutput.D_PLUS].shape
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
    metric=val.MetricMAVM(),
    labels=valplot.ValPlotLabels(x_label="Temperature [degC]"),
)
fig.savefig(output_path / "temperature_cdfs.png", bbox_inches="tight")

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
background_time_idx = len(sim_data.time) - 1

pv_plot: pv.Plotter = valvis.plot_val_metric_on_sim(
    val_result,
    temperature_tag,
    val.EMetricType.AVM,
    sim_data=sim_data,
    sim_field="temperature",
    sim_time_idx=background_time_idx,
    descriptor=sens.SensorDescriptor(
        name="Temperature", symbol=r"T", units="degC", tag="TC"
    ),
    sim_coord_units="mm",
    vis_opts=valvis.create_val_scene_opts(interactive=SHOW_PLOTS),
    metric_opts=valvis.ValMetricVisOpts(labels_always_visible=True),
)

pv_plot.screenshot(output_path / "temperature_avm_on_sim.png")

if SHOW_PLOTS:
    pv_plot.show()
pv_plot.close()


# %%
# Figures are saved even when interactive plotting is disabled. Keeping the
# complete analysis in its own directory prevents clashes with other examples.
if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
