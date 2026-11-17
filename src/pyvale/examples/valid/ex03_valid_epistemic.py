# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Sampled epistemic plate validation
================================================================================

Build on the steady plate comparison by retaining two alternative model
parameter settings. Each epistemic scenario contains its own repeated samples;
learn how to select a scenario and how envelopes summarise their spread.
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

# %%
matplotlib.use("Agg")

SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")

output_path = Path.cwd() / "pyvale-output" / "valid_ex3_epistemic"
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
with np.load(dataset.valid_data_dir() / "plate_validation_2d_steady.npz") as saved:
    coords: np.ndarray = saved["coords"].copy()
    sim_temp: np.ndarray = saved["sim_temp"].copy()  # (12, 1, 2, 6)
    exp_temp: np.ndarray = saved["exp_temp"].copy()  # (12, 1, 6)
    sim_strain: np.ndarray = saved["sim_strain"].copy()
    exp_strain: np.ndarray = saved["exp_strain"].copy()

# %%
# A shared observables registers units, component order and physical coordinates.
# Validation compares corresponding physical cells, retaining distinct point,
# component, time and repeat-experiment axes.
temperature_tag = "plate_temperature"
strain_tag = "plate_strain"
observables = {
    temperature_tag: valid.ValidObservable(
        components=("temperature",),
        geometry=valid.EValidGeometry.POINT,
        coords=coords,
        phys_units="degC",
        coord_units="mm",
    ),
    strain_tag: valid.ValidObservable(
        components=("strain_xx", "strain_yy", "strain_xy"),
        geometry=valid.EValidGeometry.POINT,
        coords=coords,
        phys_units="1",
        coord_units="mm",
    ),
}

# EPISTEMIC_TRANS holds two alternative model parameter settings: nominal
# then perturbed. Each setting contains its own six aleatory observations.
# Physical axes identify each comparison cell; ALEATORY holds its repeated
# observations. The declared axis order matches the arrays in the archive.

sim_axes = (
    valid.EValidAxis.POINT,
    valid.EValidAxis.COMPONENT,
    valid.EValidAxis.EPISTEMIC_TRANS,
    valid.EValidAxis.ALEATORY,
)

exp_axes = (
    valid.EValidAxis.POINT,
    valid.EValidAxis.COMPONENT,
    valid.EValidAxis.ALEATORY,
)

sim_arrays: dict[str, valid.ValidArray] = {}
exp_arrays: dict[str, valid.ValidArray] = {}

for observable_tag, sim_values, exp_values in (
    (temperature_tag, sim_temp, exp_temp),
    (strain_tag, sim_strain, exp_strain),
):
    sim_arrays[observable_tag] = valid.ValidArray(sim_values, sim_axes)
    exp_arrays[observable_tag] = valid.ValidArray(exp_values, exp_axes)

valid_case = valid.ValidCase(observables, sim_arrays, exp_arrays)

# %%
# 2. Run validation analysis
# --------------------------
# AVM gives a distance in the observable's units. MAVM additionally separates
# positive and negative support mismatch; INCLUDE retains repeated observations.
valid_result: valid.ValidResult = valid.ValidEngine(
    [valid.MetricAVM(), valid.MetricMAVM()]
).run(valid_case)

# %%
# 3. Post-processing and inspect data structures
# ----------------------------------------------
# metric_field.axes names each array dimension. The first three axes are always
# (point, component, time), including singleton time for steady comparisons.
# Non-singleton epistemic/offset axes follow in the engine's canonical order.
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
# ------------------------------
# Use the SensorSim appearance options so fonts, resolution and line styles
# agree with sensor figures. Each panel compares the full selected sim and exp
# epistemic envelopes, or one ECDF when only one scenario is available.
point_idxs = tuple(range(len(coords)))
fig: Figure
axes: np.ndarray
fig, axes = validplot.plot_sim_exp_cdfs(
    valid_case,
    temperature_tag,
    point_idxs=point_idxs,
    valid_plot_opts=validplot.ValidPlotOpts(x_label="Temperature [degC]"),
)
fig.savefig(output_path / "temperature_cdfs.png", bbox_inches="tight")

# %%
# Envelopes describe all retained scenarios at a site. They are pointwise ECDF
# bounds over the conditional distributions. Show every plate sensor.
# point_idxs=None plots every sensor. Supply a tuple to choose points in order,
# and set rows and/or columns to control the automatically sized landscape grid.
fig, axes = validplot.plot_sim_ecdfs(
    valid_case,
    temperature_tag,
    valid_plot_opts=validplot.ValidPlotOpts(x_label="Temperature [degC]"),
    point_idxs=None,
    offset=None,
)
fig.savefig(output_path / "temperature_envelopes.png", bbox_inches="tight")

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
assert sim_data.time is not None
background_time_idx = sim_data.time.size - 1
# Show the MAVM d- output directly, with a nonnegative colour scale.
# The d+ output is stored separately and can be selected in the same way.
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
