# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Scalar validation metrics
================================================================================

Compare six validation metrics on a scalar example with a known shift.
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

output_path = Path.cwd() / "pyvale-output" / "valid_ex01a_scalar_metrics"
output_path.mkdir(parents=True, exist_ok=True)
fig_nums = count(1)

# %%
# 1. Create scalar samples
# ------------------------
#
# Each array contains three aleatory observations. The experiment is shifted
# by two units, giving a known discrepancy for checking how the metrics
# respond.
sim_data = np.array([0.0, 1.0, 2.0])
exp = sim_data + 2.0

# %%
# 2. Build a validation case
# --------------------------
#
# A ``ValidObservable`` describes the quantity, components, geometry, and
# units. ``GLOBAL`` describes a system-level quantity. A ``ValidArray`` pairs
# a NumPy array with axis names in dimension order. ``ValidCase`` joins the
# observable and its simulation and experiment arrays by their shared tag.
scalar_observable = vl.ValidObservable(
    components=("scalar",),
    geometry=vl.EValidGeometry.GLOBAL,
    phys_units="unit",
)

observable_tag = "system_response"
observables = {observable_tag: scalar_observable}

# %%
sample_axes = (vl.EValidAxis.ALEATORY,)
sim_response = vl.ValidArray(sim_data, sample_axes)
exp_response = vl.ValidArray(exp, sample_axes)

sim_arrays = {observable_tag: sim_response}
exp_arrays = {observable_tag: exp_response}

valid_case = vl.ValidCase(observables, sim_arrays, exp_arrays)

# %%
# 3. Run the metrics
# ------------------
#
# AVM and MAVM are area validation metrics. Kolmogorov–Smirnov (KS) and
# Cramér–von Mises (CVM) also compare empirical distributions. RMSE and relative
# error use samples paired by index. Here the AVM and RMSE both equal the
# two-unit shift.
valid_result: vl.ValidResult = vl.ValidEngine(
    [
        vl.MetricAVM(),
        vl.MetricKS(),
        vl.MetricCVM(),
        vl.MetricRMSE(),
        vl.MetricRelErr(),
        vl.MetricMAVM(),
    ]
).run(valid_case)

print(80*"-")
for mm in (
    vl.MetricAVM(),
    vl.MetricKS(),
    vl.MetricCVM(),
    vl.MetricRMSE(),
    vl.MetricRelErr(),
    vl.MetricMAVM(),
):
    print(mm.get_kind().name, mm.get_scale().name, mm.get_comparison().name)
print(80*"-")

# %%
# 4. Read metric fields
# ---------------------
#
# Look up a ``MetricField`` in ``ValidResult.metrics`` using the observable tag
# followed by an ``EMetricType`` key. Read ``field.axes`` for dimension order and
# ``field.metric_vals[output]`` for the NumPy array corresponding to an
# ``EMetricOut`` key. MAVM has separate ``D_PLUS`` and ``D_MINUS`` arrays. The
# sample axis is reduced by each metric.
metrics: dict[vl.EMetricType, vl.MetricField] = valid_result.metrics[
    observable_tag
]

print("Observable tags:", tuple(observables))
print("Simulation axes and shape:", sim_response.axes, sim_response.data.shape)
print("Experiment axes and shape:", exp_response.axes, exp_response.data.shape)
print(80*"-")
for tt, ff in metrics.items():
    print(tt.name, ff.axes, ff.metric_vals)
print(80*"-")

# %%
# 5. Plot the empirical distributions
# -----------------------------------
#
# An empirical cumulative distribution function (ECDF) gives the fraction of
# samples at or below a value. ``ECDFPlot`` holds the figure, axes, and data
# selections. Pass it to the overlay functions and use its ``figure``
# attribute to save the result.
ecdf_plot: vl.ECDFPlot = vl.plot_sim_exp_cdfs(
    valid_case,
    observable_tag,
    valid_plot_opts=vl.ValidPlotOpts(x_label="Observable [unit]"),
)
fig_file = f"val_ex01_fig{next(fig_nums):02d}_observable_cdfs.png"
ecdf_plot.figure.savefig(output_path / fig_file, bbox_inches="tight")

# %%
# .. image:: /_static/val_ex01_fig01_observable_cdfs.png
#    :alt: ECDFs for the scalar simulation and its shifted experiment
#    :width: 75%
#    :align: center

# %%
# 6. Show metric overlays
# -----------------------
#
# The overlays show each metric in the context of the same ECDFs. The
# normalised AVM axis uses two units as the reference length. Integration-area
# views highlight where the directional or total discrepancy accumulates.
for mm in (
    vl.MetricAVM(),
    vl.MetricKS(),
    vl.MetricCVM(),
    vl.MetricRMSE(),
    vl.MetricRelErr(),
):
    overlay_plot: vl.ECDFPlot = vl.plot_sim_exp_cdfs(
        valid_case, observable_tag
    )
    vl.add_metric_overlay(overlay_plot, mm)

    if mm.get_kind() is vl.EMetricType.AVM:
        vl.add_normalised_ecdf_axis(overlay_plot, 2.0)

    fig_file = (
        f"val_ex01_fig{next(fig_nums):02d}_"
        f"{mm.get_kind().name.lower()}_overlay.png"
    )
    overlay_plot.figure.savefig(output_path / fig_file, bbox_inches="tight")

    if not SHOW_PLOTS:
        plt.close(overlay_plot.figure)

area_plot: vl.ECDFPlot = vl.plot_sim_exp_cdfs(
    valid_case, observable_tag
)
vl.add_metric_overlay(
    area_plot, vl.MetricAVM(), display=vl.EMetricDisp.INTEGRATION_AREA
)
fig_file = f"val_ex01_fig{next(fig_nums):02d}_avm_area.png"
area_plot.figure.savefig(output_path / fig_file, bbox_inches="tight")

if not SHOW_PLOTS:
    plt.close(area_plot.figure)

for oo, dd in (
    (vl.EMetricOut.D_PLUS, "d_plus"),
    (vl.EMetricOut.D_MINUS, "d_minus"),
):
    area_plot = vl.plot_sim_exp_cdfs(valid_case, observable_tag)

    vl.add_metric_overlay(
        area_plot,
        vl.MetricMAVM(),
        output=oo,
        display=vl.EMetricDisp.INTEGRATION_AREA,
    )

    fig_file = f"val_ex01_fig{next(fig_nums):02d}_mavm_{dd}_area.png"
    area_plot.figure.savefig(output_path / fig_file, bbox_inches="tight")

    if not SHOW_PLOTS:
        plt.close(area_plot.figure)

# %%
# .. image:: /_static/val_ex01_fig03_ks_overlay.png
#    :alt: Kolmogorov–Smirnov gap between the ECDFs
#    :width: 75%
#    :align: center
#
# .. image:: /_static/val_ex01_fig05_rmse_overlay.png
#    :alt: RMSE for samples paired by index
#    :width: 75%
#    :align: center
#
# .. image:: /_static/val_ex01_fig08_mavm_d_plus_area.png
#    :alt: Integration area for MAVM d+
#    :width: 75%
#    :align: center

# %%
# 7. Show the MAVM support span
# -----------------------------
#
# The support overlay displays the band between the lower simulation support
# shifted by d− and the upper simulation support shifted by d+.
supports = vl.find_metric_epis_supports(
    valid_case, valid_result, observable_tag, vl.EMetricType.MAVM
)

ecdf_plot = vl.plot_sim_exp_cdfs(valid_case, observable_tag)
vl.add_support_overlay(ecdf_plot, supports)

fig_file = f"val_ex01_fig{next(fig_nums):02d}_mavm_support_envelope.png"
ecdf_plot.figure.savefig(output_path / fig_file, bbox_inches="tight")

if not SHOW_PLOTS:
    plt.close(ecdf_plot.figure)

# %%
# .. image:: /_static/val_ex01_fig10_mavm_support_envelope.png
#    :alt: Full MAVM support span
#    :width: 75%
#    :align: center

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
