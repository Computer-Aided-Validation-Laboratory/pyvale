# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Transient plate validation
================================================================================

Introduce physical time as a separate comparison axis. The same 12 sensors
observe temperature and strain at five times, with six repeats at each time.
Learn to follow metric histories and select a matching simulation snapshot.
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
# Console inspection is printed below. Enable interactive windows with SHOW_PLOTS.
SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")
output_path = Path.cwd() / "pyvale-output" / "valid_ex5_transient"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Load and transform data
# -----------------------------------------
# The plate has 12 sensor locations on a 4 by 3 grid: x=10..90 mm,
# y=10..40 mm, z=0. Each site has a thermocouple (one temperature component)
# and an in-plane strain rosette (xx, yy, xy). Six virtual repeat experiments
# supply aleatory samples at five common physical times.
# The second simulation has conductivity and thermal expansion reduced by 10%.
# We treat its noisy measurements as the experiment to expose model discrepancy.
# Prepared arrays already follow the validation axis order declared below.
# Temperature has one component; strain has three components (xx, yy, xy).
with np.load(dataset.valid_data_dir() / "plate_validation_2d_nominal.npz") as saved:
    coords: np.ndarray = saved["coords"].copy()
    times: np.ndarray = saved["times"].copy()
    sim_temp: np.ndarray = saved["sim_temp"].copy()  # (12, 1, 5, 6)
    exp_temp: np.ndarray = saved["exp_temp"].copy()  # (12, 1, 5, 6)
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

# Physical axes identify each comparison cell; ALEATORY holds its repeated
# observations. The declared axis order matches the arrays in the archive.
sim_axes = (
    valid.EValidAxis.POINT,
    valid.EValidAxis.COMPONENT,
    valid.EValidAxis.TIME,
    valid.EValidAxis.ALEATORY,
)
exp_axes = (
    valid.EValidAxis.POINT,
    valid.EValidAxis.COMPONENT,
    valid.EValidAxis.TIME,
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
    time_idx=4,
    valid_plot_opts=validplot.ValidPlotOpts(x_label="Temperature [degC]"),
)
fig.savefig(output_path / "temperature_cdfs.png", bbox_inches="tight")

# %%
# Each panel follows one physical sensor through time. Explicit selectors fix
# any remaining epistemic scenario; d+ identifies the directional MAVM quantity.
for metric_output, direction in (
    (valid.EMetricOutput.D_PLUS, "d_plus"),
    (valid.EMetricOutput.D_MINUS, "d_minus"),
):
    fig, axes = validplot.plot_valid_metric_time_points(
        valid_result,
        temperature_tag,
        valid.EMetricType.MAVM,
        point_idxs=point_idxs,
        output=metric_output,
    )
    fig.savefig(output_path / f"temperature_{direction}_time.png", bbox_inches="tight")

# %%
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
            valid.EValidResAxis.TIME: 4,
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
