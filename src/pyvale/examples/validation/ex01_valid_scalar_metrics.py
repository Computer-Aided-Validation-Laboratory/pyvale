# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Scalar validation metrics
================================================================================
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyvale.validation as vl

matplotlib.use("Agg")

# %%
SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")

output_path = Path.cwd() / "pyvale-output" / "valid_ex01_scalar_metrics"
output_path.mkdir(parents=True, exist_ok=True)

# %%
sim_data = np.array([0.0, 1.0, 2.0])
exp = sim_data + 2.0

# %%
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
for metric in (
    vl.MetricAVM(),
    vl.MetricKS(),
    vl.MetricCVM(),
    vl.MetricRMSE(),
    vl.MetricRelErr(),
    vl.MetricMAVM(),
):
    print(metric.get_kind().name, metric.get_scale().name, metric.get_comparison().name)
print(80*"-")

# %%
metrics: dict[vl.EMetricType, vl.MetricField] = valid_result.metrics[
    observable_tag
]

print("Observable tags:", tuple(observables))
print("Simulation axes and shape:", sim_response.axes, sim_response.data.shape)
print("Experiment axes and shape:", exp_response.axes, exp_response.data.shape)
print(80*"-")
for metric_type, metric_field in metrics.items():
    print(metric_type.name, metric_field.axes, metric_field.outputs)
print(80*"-")

# %%
ecdf_plot: vl.ECDFPlot = vl.plot_sim_exp_cdfs(
    valid_case,
    observable_tag,
    valid_plot_opts=vl.ValidPlotOpts(x_label="Observable [unit]"),
)
ecdf_plot.figure.savefig(output_path / "observable_cdfs.png", bbox_inches="tight")

# %%
for metric in (
    vl.MetricAVM(),
    vl.MetricKS(),
    vl.MetricCVM(),
    vl.MetricRMSE(),
    vl.MetricRelErr(),
):
    overlay_plot: vl.ECDFPlot = vl.plot_sim_exp_cdfs(
        valid_case, observable_tag
    )
    vl.add_metric_overlay(overlay_plot, metric)
    if metric.get_kind() is vl.EMetricType.AVM:
        vl.add_normalised_ecdf_axis(overlay_plot, 2.0)
    overlay_plot.figure.savefig(
        output_path / f"{metric.get_kind().name.lower()}_overlay.png",
        bbox_inches="tight",
    )
    if not SHOW_PLOTS:
        plt.close(overlay_plot.figure)

area_plot: vl.ECDFPlot = vl.plot_sim_exp_cdfs(
    valid_case, observable_tag
)
vl.add_metric_overlay(
    area_plot, vl.MetricAVM(), display=vl.EMetricDisp.INTEGRATION_AREA
)
area_plot.figure.savefig(output_path / "avm_area.png", bbox_inches="tight")
if not SHOW_PLOTS:
    plt.close(area_plot.figure)

for metric_output, direction in (
    (vl.EMetricOut.D_PLUS, "d_plus"),
    (vl.EMetricOut.D_MINUS, "d_minus"),
):
    area_plot = vl.plot_sim_exp_cdfs(valid_case, observable_tag)
    vl.add_metric_overlay(
        area_plot,
        vl.MetricMAVM(),
        output=metric_output,
        display=vl.EMetricDisp.INTEGRATION_AREA,
    )
    area_plot.figure.savefig(
        output_path / f"mavm_{direction}_area.png", bbox_inches="tight"
    )
    if not SHOW_PLOTS:
        plt.close(area_plot.figure)

model_cdf: vl.ValidECDF = vl.calc_ecdf(sim_data)
fig, _ = vl.plot_u_pooling_cdf([model_cdf] * exp.size, exp)
fig.savefig(output_path / "u_pooling.png", bbox_inches="tight")
if not SHOW_PLOTS:
    plt.close(fig)

supports = vl.find_metric_epis_supports(
    valid_case, valid_result, observable_tag, vl.EMetricType.MAVM
)
ecdf_plot = vl.plot_sim_exp_cdfs(valid_case, observable_tag)
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
