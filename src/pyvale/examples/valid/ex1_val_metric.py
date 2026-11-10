# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Analytic validation metrics
================================================================================

A simple validation example with with three observations so each distribution can be
checked by hand.
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
# 1. Load and transform data (pre-processing)
# -----------------------------------------
# Each (3,) array is one scalar observable at one point and physical time.
# Entries are repeat observations, not sensor coordinates or time samples.
simulation = np.array([0.0, 1.0, 2.0])
experiment = simulation + 2.0
geometry = val.ValGeometry(val.EValGeometry.GLOBAL, None)
domain = val.ValDomain(
    {
        "observable": val.ValObservable(
            geometry,
            val.EValOperator.SAMPLE,
            ("scalar",),
            None,
            "unit",
        )
    }
)
val_case = val.ValCase(
    domain,
    val.ValData({"observable": val.ValArray(simulation, (val.EValAxis.ALEATORY,))}),
    val.ValData({"observable": val.ValArray(experiment, (val.EValAxis.ALEATORY,))}),
)

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
# type; do not interpret the three observation entries as three physical points.
metric_fields = result.fields["observable"]
if VERBOSE:
    for metric_type, field in metric_fields.items():
        print(metric_type.name, field.axes, field.metric)

# %%
# 4. Analysis and visualisation
# ------------------------------
# Both source distributions appear in the legend. The title identifies the two
# directional distances rather than leaving "MAVM" ambiguous.
fig, ax = val.plot_val_cdfs(
    val_case,
    result,
    "observable",
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
