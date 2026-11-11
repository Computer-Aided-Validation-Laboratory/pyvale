# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Analytic validation metrics
================================================================================

Build a validation case from three scalar observations, then inspect its
metrics and empirical distributions. Each data-model object is introduced below.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyvale.sensorsim as sens
import pyvale.valid as val

matplotlib.use("Agg")

# %%
# Interactive windows and console inspection are opt-in. Change the backend to
# a GUI backend when enabling SHOW_PLOTS outside a notebook.
SHOW_PLOTS = False
VERBOSE = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")
output_path = Path.cwd() / "pyvale-output" / "valid_ex1_metric"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Load and transform data
# -----------------------------------------
# The three entries represent three independent observations for a single
# point in the domain. Their only varying axis is ALEATORY (repeat sampling).
simulation = np.array([0.0, 1.0, 2.0])
experiment = simulation + 2.0

# %%
# ValGeometry describes the physical support and coordinate frame. GLOBAL
# represents a quantity associated with the whole system. A spatial sensor
# array uses POINT geometry with coordinates shaped (n_points, 3), as in ex2.
geometry = val.ValGeometry(val.EValGeometry.GLOBAL, None)

# ValObservable describes how a quantity is observed: geometry, measurement
# operator, component order, physical times and units. SAMPLE denotes values
# sampled on that geometry. Here we have one scalar component at one state.
scalar_observable = val.ValObservable(
    geometry, val.EValOperator.SAMPLE, ("scalar",), None, "unit"
)

# ValDomain collects the definitions shared by simulation and experiment.
# The tag is a user-chosen identifier for this comparison. Component names
# describe entries within an observable; the tag identifies the observable.
observable_tag = "system_response"
domain = val.ValDomain({observable_tag: scalar_observable})

# %%
# ValArray associates numerical samples with named axes. This separates the
# physical comparison cells from the uncertainty samples within each cell.
# Omitted physical axes have length one; the sample arrays here have shape (3,).
sample_axes = (val.EValAxis.ALEATORY,)
sim_response = val.ValArray(simulation, sample_axes)
exp_response = val.ValArray(experiment, sample_axes)

# ValData collects one source's quantities under their observable tags. Keeping
# source samples separate from the shared domain lets each source have its own
# repeat count and uncertainty scenarios while describing the same comparison.
sim_val_data = val.ValData({observable_tag: sim_response})
exp_val_data = val.ValData({observable_tag: exp_response})

# ValCase joins the domain and both sources. It checks matching tag sets and
# physical axis lengths. Prepare source values in the domain's component order,
# physical locations, times and units. The engine uses this SAME tag to find
# each observable in domain.observables, sim.quantities and exp.quantities.
val_case = val.ValCase(domain, sim_val_data, exp_val_data)

# %%
# 2. Run validation analysis
# --------------------------
# AVM is 2 units, KS is 2/3 (one support value overlaps), and paired RMSE is 2.
# MAVM also accounts for experimental confidence bounds and reports d+ and d−.
mavm = val.MetricMAVM()
result = val.ValidationEngine(
    [
        val.MetricAVM(),
        val.MetricKS(),
        val.MetricCVM(),
        val.MetricRMSE(),
        mavm,
    ]
).run(val_case)

# %%
# 3. Post-processing and inspect data structures
# ----------------------------------------------
# Engine arrays have shape (1 point, 1 component, 1 time). Select by metric
# type. The three input entries remain independent observations for one point.
metric_fields = result.fields[observable_tag]
if VERBOSE:
    print("Observable tags:", tuple(domain.observables))
    print("Simulation axes and shape:", sim_response.axes, sim_response.data.shape)
    print("Experiment axes and shape:", exp_response.axes, exp_response.data.shape)
    for metric_type, field in metric_fields.items():
        print(metric_type.name, field.axes, field.metric)

# %%
# 4. Analysis and visualisation
# ------------------------------
# Both source distributions appear in the legend. The title identifies the two
# directional distances d+ and d- alongside the total MAVM.
fig, ax = val.plot_val_cdfs(
    val_case,
    result,
    observable_tag,
    metric=mavm,
    plot_opts=sens.PlotOptsGeneral(),
    labels=val.ValPlotLabels(x_label="Observable [unit]"),
)
fig.savefig(output_path / "observable_cdfs.png", bbox_inches="tight")

# %%
# Figures are saved even when interactive plotting is disabled. Keeping the
# complete analysis in its own directory prevents clashes with other examples.
if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
