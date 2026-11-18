# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Steady transferable/nuisance ECDF views on both simulation and experiment.

Sim has (3 transferable, 3 nuisance, 5 aleatory) samples; exp has (2,3,4).
Nuisance alternatives have asymmetric shifts, making mean, median and quartile
summaries visibly different. Their quantiles are not probabilities assigned to
nuisance parameters. Ex06 selects and ranks actual nuisance cases instead.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyvale.validation as vl

matplotlib.use("Agg")
SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")
output_path = Path.cwd() / "pyvale-output" / "valid_ex5_epistemic_axes"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Prepare positive temperatures at two sites and conditional sample grids.
# One sim nuisance alternative has an 18 degC shift, exposing mean sensitivity.
coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
baseline = 12.0 + 2.0 * coords[:, 0]
sim_shifts = 0.2 * np.array([[1.0, 90.0, 8.0], [2.0, 3.0, 7.0], [4.0, 6.0, 5.0]])
exp_shifts = 0.2 * np.array([[0.0, 1.0, 7.0], [3.0, 4.0, 8.0]])
sim_samples = (
    baseline[:, None, None, None, None]
    + sim_shifts[None, None, :, :, None]
    + np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
)
exp_samples = (
    baseline[:, None, None, None, None]
    + exp_shifts[None, None, :, :, None]
    + np.array([-0.8, -0.2, 0.2, 0.8])
)
axes = (
    vl.EValidAxis.POINT,
    vl.EValidAxis.COMPONENT,
    vl.EValidAxis.EPISTEMIC_TRANS,
    vl.EValidAxis.EPISTEMIC_NUIS,
    vl.EValidAxis.ALEATORY,
)
observable_tag = "temperature"
valid_case = vl.ValidCase(
    {
        observable_tag: vl.ValidObservable(
            components=("temperature",),
            geometry=vl.EValidGeometry.POINT,
            coords=coords,
            phys_units="degC",
            coord_units="m",
        )
    },
    {observable_tag: vl.ValidArray(sim_samples, axes)},
    {observable_tag: vl.ValidArray(exp_samples, axes)},
)
valid_result: vl.ValidResult = vl.ValidEngine(
    [vl.MetricAVM(), vl.MetricMAVM()]
).run(valid_case)
print("Result axes:", valid_result.metrics[observable_tag][vl.EMetricType.MAVM].axes)

# %%
# 2. Numerical bounds resolve nuisance first, then envelope transferable cases.
# KEEP envelopes every original joint case. A singleton nuis_idxs conditions on
# an actual sample; MEAN/MEDIAN/QUANTILE form synthetic quantile curves.
for name, reduction, probability in (
    ("all", vl.EValidECDFReduce.KEEP, None),
    ("mean", vl.EValidECDFReduce.MEAN, None),
    ("median", vl.EValidECDFReduce.MEDIAN, None),
    ("minimum", vl.EValidECDFReduce.QUANTILE, 0.0),
    ("lower_quartile", vl.EValidECDFReduce.QUANTILE, 0.25),
    ("upper_quartile", vl.EValidECDFReduce.QUANTILE, 0.75),
    ("maximum", vl.EValidECDFReduce.QUANTILE, 1.0),
):
    bounds: vl.ValidECDFBounds = vl.find_ecdf_epis_bounds(
        valid_case.sim[observable_tag], nuis_reduce=reduction, nuis_quantile=probability
    )
    np.savetxt(
        output_path / f"{name}_bounds.csv",
        np.column_stack((bounds.probabilities, bounds.lower, bounds.upper)),
        delimiter=",",
        header="probability,lower,upper",
        comments="",
    )

# 3. A view controls which summaries and original curves appear. Pointwise
# envelopes can switch controlling cases along p; they need not be two actual
# scenario ECDFs. The default uses all cases on both axes, without pooling.
views = (
    ("outer_outer", vl.ValidECDFView(), vl.ValidECDFView()),
    (
        "fixed_nuisance",
        vl.ValidECDFView(nuis_idxs=(1,)),
        vl.ValidECDFView(nuis_idxs=(1,)),
    ),
    (
        "subset",
        vl.ValidECDFView(trans_idxs=(0, 2), nuis_idxs=(0, 2)),
        vl.ValidECDFView(trans_idxs=(0,), nuis_idxs=(0, 2)),
    ),
    (
        "median_nuisance",
        vl.ValidECDFView(nuis_reduce=vl.EValidECDFReduce.MEDIAN),
        vl.ValidECDFView(nuis_reduce=vl.EValidECDFReduce.MEDIAN),
    ),
    (
        "lower_quartile",
        vl.ValidECDFView(
            nuis_reduce=vl.EValidECDFReduce.QUANTILE, nuis_quantile=0.25
        ),
        vl.ValidECDFView(
            nuis_reduce=vl.EValidECDFReduce.QUANTILE, nuis_quantile=0.25
        ),
    ),
    (
        "upper_quartile",
        vl.ValidECDFView(
            nuis_reduce=vl.EValidECDFReduce.QUANTILE, nuis_quantile=0.75
        ),
        vl.ValidECDFView(
            nuis_reduce=vl.EValidECDFReduce.QUANTILE, nuis_quantile=0.75
        ),
    ),
    (
        "mean_curves",
        vl.ValidECDFView(
            nuis_reduce=vl.EValidECDFReduce.MEAN,
            trans_reduce=vl.EValidECDFReduce.MEAN,
        ),
        vl.ValidECDFView(
            nuis_reduce=vl.EValidECDFReduce.MEAN,
            trans_reduce=vl.EValidECDFReduce.MEAN,
        ),
    ),
    (
        "median_curves",
        vl.ValidECDFView(
            nuis_reduce=vl.EValidECDFReduce.MEDIAN,
            trans_reduce=vl.EValidECDFReduce.MEDIAN,
        ),
        vl.ValidECDFView(
            nuis_reduce=vl.EValidECDFReduce.MEDIAN,
            trans_reduce=vl.EValidECDFReduce.MEDIAN,
        ),
    ),
    (
        "selected_scenarios",
        vl.ValidECDFView(bounds=False, scenario_idxs=((0, 0), (2, 2))),
        vl.ValidECDFView(bounds=False, scenario_idxs=((0, 1), (1, 2))),
    ),
    (
        "independent_views",
        vl.ValidECDFView(nuis_reduce=vl.EValidECDFReduce.MEDIAN),
        vl.ValidECDFView(
            nuis_reduce=vl.EValidECDFReduce.MEAN,
            trans_reduce=vl.EValidECDFReduce.MEDIAN,
        ),
    ),
)
for name, sim_view, exp_view in views:
    ecdf_plot: vl.ECDFPlot = vl.plot_sim_exp_cdfs(
        valid_case,
        observable_tag,
        point_idxs=None,
        sim_view=sim_view,
        exp_view=exp_view,
    )
    ecdf_plot.figure.savefig(output_path / f"{name}.png", bbox_inches="tight")
    if not SHOW_PLOTS:
        plt.close(ecdf_plot.figure)
sim_plot: vl.ECDFPlot = vl.plot_sim_ecdfs(
    valid_case,
    observable_tag,
    point_idxs=None,
    view=vl.ValidECDFView(nuis_reduce=vl.EValidECDFReduce.MEDIAN),
)
sim_plot.figure.savefig(output_path / "sim_median_nuisance.png", bbox_inches="tight")
exp_plot: vl.ECDFPlot = vl.plot_exp_ecdfs(
    valid_case, observable_tag, point_idxs=None
)
exp_plot.figure.savefig(output_path / "exp_envelopes.png", bbox_inches="tight")

# %%
# 4. Normalisation uses one reference constant per panel: the mean sim quantile
# at p=.5 or p=.9. The physical axis remains below; the dimensionless axis above
# is x/reference, not a probability-dependent displacement of each curve.
for probability in (0.5, 0.9):
    ecdf_plot = vl.plot_sim_exp_cdfs(valid_case, observable_tag, point_idxs=None)
    vl.add_normalised_ecdf_axis(
        ecdf_plot, vl.ValidECDFReference(probability=probability)
    )
    vl.add_metric_overlay(ecdf_plot, vl.MetricAVM())
    ecdf_plot.figure.savefig(
        output_path / f"avm_normalised_p{probability:g}.png", bbox_inches="tight"
    )
    if not SHOW_PLOTS:
        plt.close(ecdf_plot.figure)
ks_plot: vl.ECDFPlot = vl.plot_sim_exp_cdfs(
    valid_case, observable_tag, point_idxs=None
)
vl.add_metric_overlay(ks_plot, vl.MetricKS())
ks_plot.figure.savefig(output_path / "ks_bound_distances.png", bbox_inches="tight")

# Support shifts search original conditional comparisons. Integration areas
# require selecting an actual sim/exp pair; summaries are not pseudo-observations.
for metric_output, direction in (
    (vl.EMetricOut.D_PLUS, "d_plus"),
    (vl.EMetricOut.D_MINUS, "d_minus"),
):
    support_plot: vl.ECDFPlot = vl.plot_sim_exp_cdfs(
        valid_case, observable_tag, point_idxs=None
    )
    vl.add_metric_overlay(support_plot, vl.MetricMAVM(), output=metric_output)
    support_plot.figure.savefig(
        output_path / f"mavm_{direction}_support.png", bbox_inches="tight"
    )
    area_plot: vl.ECDFPlot = vl.plot_sim_exp_cdfs(
        valid_case,
        observable_tag,
        point_idxs=None,
        sim_view=vl.ValidECDFView(trans_idxs=(1,), nuis_idxs=(1,)),
        exp_view=vl.ValidECDFView(trans_idxs=(0,), nuis_idxs=(1,)),
    )
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
        plt.close(support_plot.figure)
        plt.close(area_plot.figure)
if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
