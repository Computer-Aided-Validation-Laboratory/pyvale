# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Coherent epistemic measurement offsets
================================================================================

Extend the steady epistemic comparison with asymmetric calibration bounds.
Learn how coherent offsets describe uncertainty shared by every repeat at a
measurement location, and how to select nominal, lower and upper comparisons.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyvale.data as dataset
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
output_path = Path.cwd() / "pyvale-output" / "valid_ex4_bounds"
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
    coords = saved["coords"].copy()
    sim_temp = saved["sim_temp"].copy()  # (12, 1, 2, 6)
    exp_temp = saved["exp_temp"].copy()  # (12, 1, 6)
    sim_strain = saved["sim_strain"].copy()
    exp_strain = saved["exp_strain"].copy()

# %%
# A shared domain registers units, component order and physical coordinates.
# Validation compares corresponding physical cells, retaining distinct point,
# component, time and repeat-experiment axes.
geometry = val.ValGeometry(
    val.EValGeometry.POINT, coords, coord_units="mm", coord_frame="plate"
)
temperature_tag = "plate_temperature"
strain_tag = "plate_strain"
domain = val.ValDomain(
    {
        temperature_tag: val.ValObservable(
            geometry=geometry,
            operator=val.EValOperator.SAMPLE,
            components=("temperature",),
            times=None,
            phys_units="degC",
            time_units="",
        ),
        strain_tag: val.ValObservable(
            geometry=geometry,
            operator=val.EValOperator.SAMPLE,
            components=("strain_xx", "strain_yy", "strain_xy"),
            times=None,
            phys_units="1",
            time_units="",
        ),
    }
)

# EPISTEMIC_TRANS holds two alternative model parameter settings: nominal
# then perturbed. Each setting contains its own six aleatory observations.
# Physical axes identify each comparison cell; ALEATORY holds its repeated
# observations. The declared axis order matches the arrays in the archive.
sim_axes = (
    val.EValAxis.POINT,
    val.EValAxis.COMPONENT,
    val.EValAxis.EPISTEMIC_TRANS,
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
    # Lower and upper magnitudes describe a coherent calibration shift in
    # observable units, shared by every repeat at a physical comparison cell.
    magnitude = 0.5 if observable_tag == temperature_tag else 4.0e-6
    if observable_tag == temperature_tag:
        spatial = coords[:, 0] / 100.0  # (12 points,)
        bound = val.EpistemicOffsets(
            lower=magnitude * (1.0 + 0.2 * spatial),
            upper=1.5 * magnitude * (1.0 + 0.1 * spatial),
            axes=(val.EValAxis.POINT,),
        )
    else:
        bound = val.EpistemicOffsets(lower=magnitude, upper=1.5 * magnitude)
    sim_arrays[observable_tag] = val.ValArray(sim_values, sim_axes)
    exp_arrays[observable_tag] = val.ValArray(exp_values, exp_axes, offsets=bound)

val_case = val.ValCase(domain, sim_arrays, exp_arrays)

# %%
# 2. Run validation analysis
# --------------------------
# AVM gives a distance in the observable's units. MAVM additionally separates
# positive and negative support mismatch; INCLUDE retains repeated observations.
mavm = val.MetricMAVM(duplicates=val.EMAVMDuplicates.INCLUDE)
val_result = val.ValidationEngine([val.MetricAVM(), mavm]).run(val_case)

# %%
# 3. Post-processing and inspect data structures
# ----------------------------------------------
# metric_field.axes names each array dimension. The first three axes are always
# (point, component, time), including singleton time for steady comparisons.
# Non-singleton epistemic/offset axes follow in the engine's canonical order.
metric_field = val_result.get(temperature_tag, val.EMetricType.MAVM)
metric_shape = metric_field.metric.shape
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
    time_idx=0,
    metric=mavm,
    labels=valplot.ValPlotLabels(x_label="Temperature [degC]"),
)
fig.savefig(output_path / "temperature_cdfs.png", bbox_inches="tight")

# %%
# Envelopes describe all retained scenarios at a site. They are pointwise ECDF
# bounds over the conditional distributions. Show every plate sensor.
fig, axes = valplot.plot_val_cdf_envelope_points(
    val_case,
    temperature_tag,
    source=val.EValSource.EXP,
    time_idx=0,
    labels=valplot.ValPlotLabels(x_label="Temperature [degC]"),
)
fig.savefig(output_path / "temperature_envelopes.png", bbox_inches="tight")

# %%
# Overlay validation points on the nominal model in its original xyz geometry.
# The model mesh and sensor positions must share mm coordinates and the plate
# coordinate frame. Background nodal scalars (n_nodes,) have a physical colourbar.
sim_path = dataset.thermomechanical_2d_experiment_paths()[0]
sim_data = mh.ExodusLoader(sim_path).load_all_sim_data()
sens.scale_length_units(1000.0, sim_data, disp_keys=("disp_x", "disp_y"))
background_time_idx = len(sim_data.time) - 1
background_field = valvis.ValSimField(
    sim_data.node_vars["temperature"][:, background_time_idx],
    background_time_idx,
    sens.SensorDescriptor(name="Temperature", symbol=r"T", units="degC"),
)
point_labels = tuple(sens.SensorDescriptor(tag="TC").create_sensor_tags(len(coords)))
# Blue/red encode negative/positive MAVM directional balance. A white point
# can have nonzero total mismatch because d+ and d- can cancel in this display.
pv_plot = valvis.plot_val_metric_on_sim(
    val_result,
    temperature_tag,
    val.EMetricType.MAVM,
    selectors={
        val.EValResAxis.SIM_TRANS: 0,
        val.EValResAxis.EXP_OFFSET: val.EEpisOffset.UPPER,
    },
    part=val.EValMetricPart.BALANCE,
    sim_data=sim_data,
    sim_field=background_field,
    sim_coord_units="mm",
    sim_coord_frame="plate",
    vis_opts=valvis.create_val_vis_opts(interactive=True) if SHOW_PLOTS else None,
    metric_opts=valvis.ValVisOpts(
        point_labels=point_labels, labels_always_visible=True
    ),
)
pv_plot.screenshot(output_path / "temperature_mavm_balance_on_sim.png")
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
