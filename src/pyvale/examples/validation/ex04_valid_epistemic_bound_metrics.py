# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Metrics on epistemic bounds
================================================================================

Compare validation envelopes built from ECDF bounds and from original scenario
pairs.
"""

from itertools import count
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyvale.validation as vl

# %%
matplotlib.use("Agg")

SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")

output_path = Path.cwd() / "pyvale-output" / "valid_ex04_epistemic_bound_metrics"
output_path.mkdir(parents=True, exist_ok=True)
fig_nums = count(1)

# %%
# 1. Set up two epistemic families
# --------------------------------
#
# Each row is an epistemic case and each column is an aleatory sample within
# that case. Use ``EPISTEMIC`` for the transferable axis in this single-axis
# example.
observable_tag = "response"

axes = (vl.EValidAxis.EPISTEMIC, vl.EValidAxis.ALEATORY)

valid_case = vl.ValidCase(
    {
        observable_tag: vl.ValidObservable(
            components=("response",),
            geometry=vl.EValidGeometry.GLOBAL,
            phys_units="unit",
        )
    },
    {observable_tag: vl.ValidArray(np.array([[0.0, 100.0], [40.0, 60.0]]), axes)},
    {observable_tag: vl.ValidArray(np.array([[-10.0, 0.0], [100.0, 110.0]]), axes)},
)

# %%
# 2. Calculate support from ECDF bounds
# -------------------------------------
#
# Find the pointwise envelope of each ECDF family. The bound-based AVM and
# MAVM functions use those envelopes to produce ``ECDFSuppBounds``, which
# holds the support curves for plotting.
sim_bounds: vl.ValidECDFBounds = vl.find_ecdf_epis_bounds(
    valid_case.sim[observable_tag]
)

exp_bounds: vl.ValidECDFBounds = vl.find_ecdf_epis_bounds(
    valid_case.exp[observable_tag]
)

avm_bounds: vl.ECDFSuppBounds = vl.calc_avm_over_epis_bounds(
    sim_bounds, exp_bounds
)

mavm_bounds: vl.ECDFSuppBounds = vl.calc_mavm_over_epis_bounds(
    sim_bounds, exp_bounds, alpha=0.5
)

# %%
# 3. Calculate support from original cases
# ----------------------------------------
#
# Run the metrics over each simulation and experiment case pair. Use the
# resulting metric fields and the original samples to assemble the span of
# their shifted supports. Both MAVM calculations use the same significance
# level.
valid_result: vl.ValidResult = vl.ValidEngine(
    [vl.MetricAVM(), vl.MetricMAVM(alpha=0.5)]
).run(valid_case)

avm_original: vl.ECDFSuppBounds = vl.find_metric_epis_supports(
    valid_case, valid_result, observable_tag, vl.EMetricType.AVM
)

mavm_original: vl.ECDFSuppBounds = vl.find_metric_epis_supports(
    valid_case, valid_result, observable_tag, vl.EMetricType.MAVM
)

# %%
# 4. Compare the resulting spans
# ------------------------------
#
# Add each set of supports to the same ECDF comparison so the difference
# between the two calculation routes is visible.
for rr, aa, mm in (
    ("bounds", avm_bounds, mavm_bounds),
    ("original", avm_original, mavm_original),
):
    ecdf_plot: vl.ECDFPlot = vl.plot_sim_exp_cdfs(
        valid_case, observable_tag
    )

    vl.add_support_overlay(ecdf_plot, aa)
    fig_file = f"val_ex04_fig{next(fig_nums):02d}_avm_{rr}.png"
    ecdf_plot.figure.savefig(output_path / fig_file, bbox_inches="tight")

    if not SHOW_PLOTS:
        plt.close(ecdf_plot.figure)

    ecdf_plot = vl.plot_sim_exp_cdfs(valid_case, observable_tag)
    vl.add_support_overlay(
        ecdf_plot, mm
    )
    fig_file = f"val_ex04_fig{next(fig_nums):02d}_mavm_support_envelope_{rr}.png"
    ecdf_plot.figure.savefig(output_path / fig_file, bbox_inches="tight")

    if not SHOW_PLOTS:
        plt.close(ecdf_plot.figure)

if SHOW_PLOTS:
    plt.show()
