# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Analytic validation metrics
================================================================================

Build a validation case from three scalar observations, then inspect its
metrics and empirical distributions. Introduces the validation data model.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from pyvale import valid
from pyvale.valid import validplot

matplotlib.use("Agg")

# %%
# Console inspection is printed below. Enable interactive windows with SHOW_PLOTS.
SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")

output_path = Path.cwd() / "pyvale-output" / "valid_ex1_metric"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Load and transform data
# -----------------------------------------
# The three entries represent three independent observations for a single
# point in the observables. Their only varying axis is ALEATORY (repeat sampling).
sim_data = np.array([0.0, 1.0, 2.0])
exp = sim_data + 2.0

# %%
# Geometry is declared directly on ValidObservable. GLOBAL represents a quantity
# associated with the whole system, so it has no point coordinates. Spatial
# observables supply coordinates shaped (n_points, 3), as in ex02.
# ValidObservable records geometry, component order, physical times and units.
# Use an observable tag that identifies the measurement being compared.
# Here we have one scalar component at one state.
scalar_observable = valid.ValidObservable(
    components=("scalar",),
    geometry=valid.EValidGeometry.GLOBAL,
    phys_units="unit",
)

# The observable dictionary holds metadata shared by simulation and experiment.
# The tag is a user-chosen identifier for this comparison. Component names
# describe entries within an observable; the tag identifies the observable.
observable_tag = "system_response"
observables = {observable_tag: scalar_observable}

# %%
# ValidArray associates numerical samples with named axes. This separates the
# physical comparison cells from the uncertainty samples within each cell.
# Omitted physical axes have length one; the sample arrays here have shape (3,).
sample_axes = (valid.EValidAxis.ALEATORY,)
sim_response = valid.ValidArray(sim_data, sample_axes)
exp_response = valid.ValidArray(exp, sample_axes)

# Dictionaries collect each source's ValidArray quantities by observable tag. Keeping
# source samples separate from the shared observables lets each source have its own
# repeat count and uncertainty scenarios while describing the same comparison.
sim_arrays = {observable_tag: sim_response}
exp_arrays = {observable_tag: exp_response}

# ValidCase joins the observables and both sources. It checks matching tag sets and
# physical axis lengths. Prepare source values in the observables's component order,
# physical locations, times and units. The engine uses this SAME tag to find
# each observable in observables, sim and exp.
valid_case = valid.ValidCase(observables, sim_arrays, exp_arrays)

# %%
# 2. Run validation analysis
# --------------------------
# AVM is 2 units, KS is 2/3 (one support value overlaps), and paired RMSE is 2.
# MAVM also accounts for experimental confidence bounds and reports d+ and d−.
valid_result: valid.ValidResult = valid.ValidEngine(
    [
        valid.MetricAVM(),
        valid.MetricKS(),
        valid.MetricCVM(),
        valid.MetricRMSE(),
        valid.MetricMAVM(),
    ]
).run(valid_case)

# %%
# 3. Post-processing and inspect data structures
# ----------------------------------------------
# Engine arrays have shape (1 point, 1 component, 1 time). Select by metric
# type. The three input entries remain independent observations for one point.
metrics: dict[valid.EMetricType, valid.ValidMetricField] = valid_result.metrics[
    observable_tag
]
print("Observable tags:", tuple(observables))
print("Simulation axes and shape:", sim_response.axes, sim_response.data.shape)
print("Experiment axes and shape:", exp_response.axes, exp_response.data.shape)
for metric_type, metric_field in metrics.items():
    print(metric_type.name, metric_field.axes, metric_field.outputs)

# %%
# 4. Analysis and visualisation
# ------------------------------
# Both source distributions appear in the legend. Directional MAVM distances
# are plotted separately below; the ECDF comparison needs no metric result.
fig: Figure
axes: np.ndarray
fig, axes = validplot.plot_sim_exp_cdfs(
    valid_case,
    observable_tag,
    valid_plot_opts=validplot.ValidPlotOpts(x_label="Observable [unit]"),
)
fig.savefig(output_path / "observable_cdfs.png", bbox_inches="tight")

# %%
# Figures are saved even when interactive plotting is disabled. Keeping the
# complete analysis in its own directory prevents clashes with other examples.
if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
