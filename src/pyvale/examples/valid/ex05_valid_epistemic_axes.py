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

from pyvale import valid
from pyvale.valid import validplot

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
    valid.EValidAxis.POINT,
    valid.EValidAxis.COMPONENT,
    valid.EValidAxis.EPISTEMIC_TRANS,
    valid.EValidAxis.EPISTEMIC_NUIS,
    valid.EValidAxis.ALEATORY,
)
observable_tag = "temperature"
valid_case = valid.ValidCase(
    {
        observable_tag: valid.ValidObservable(
            components=("temperature",),
            geometry=valid.EValidGeometry.POINT,
            coords=coords,
            phys_units="degC",
            coord_units="m",
        )
    },
    {observable_tag: valid.ValidArray(sim_samples, axes)},
    {observable_tag: valid.ValidArray(exp_samples, axes)},
)
valid_result: valid.ValidResult = valid.ValidEngine(
    [valid.MetricAVM(), valid.MetricMAVM()]
).run(valid_case)
print("Result axes:", valid_result.metrics[observable_tag][valid.EMetricType.MAVM].axes)

# %%
# 2. Numerical bounds resolve nuisance first, then envelope transferable cases.
# KEEP envelopes every original joint case. A singleton nuis_idxs conditions on
# an actual sample; MEAN/MEDIAN/QUANTILE form synthetic quantile curves.
for name, reduction, probability in (
    ("all", valid.EValidECDFReduce.KEEP, None),
    ("mean", valid.EValidECDFReduce.MEAN, None),
    ("median", valid.EValidECDFReduce.MEDIAN, None),
    ("minimum", valid.EValidECDFReduce.QUANTILE, 0.0),
    ("lower_quartile", valid.EValidECDFReduce.QUANTILE, 0.25),
    ("upper_quartile", valid.EValidECDFReduce.QUANTILE, 0.75),
    ("maximum", valid.EValidECDFReduce.QUANTILE, 1.0),
):
    bounds: valid.ValidECDFBounds = valid.find_ecdf_epis_bounds(
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
    ("outer_outer", valid.ValidECDFView(), valid.ValidECDFView()),
    (
        "fixed_nuisance",
        valid.ValidECDFView(nuis_idxs=(1,)),
        valid.ValidECDFView(nuis_idxs=(1,)),
    ),
    (
        "subset",
        valid.ValidECDFView(trans_idxs=(0, 2), nuis_idxs=(0, 2)),
        valid.ValidECDFView(trans_idxs=(0,), nuis_idxs=(0, 2)),
    ),
    (
        "median_nuisance",
        valid.ValidECDFView(nuis_reduce=valid.EValidECDFReduce.MEDIAN),
        valid.ValidECDFView(nuis_reduce=valid.EValidECDFReduce.MEDIAN),
    ),
    (
        "lower_quartile",
        valid.ValidECDFView(
            nuis_reduce=valid.EValidECDFReduce.QUANTILE, nuis_quantile=0.25
        ),
        valid.ValidECDFView(
            nuis_reduce=valid.EValidECDFReduce.QUANTILE, nuis_quantile=0.25
        ),
    ),
    (
        "upper_quartile",
        valid.ValidECDFView(
            nuis_reduce=valid.EValidECDFReduce.QUANTILE, nuis_quantile=0.75
        ),
        valid.ValidECDFView(
            nuis_reduce=valid.EValidECDFReduce.QUANTILE, nuis_quantile=0.75
        ),
    ),
    (
        "mean_curves",
        valid.ValidECDFView(
            nuis_reduce=valid.EValidECDFReduce.MEAN,
            trans_reduce=valid.EValidECDFReduce.MEAN,
        ),
        valid.ValidECDFView(
            nuis_reduce=valid.EValidECDFReduce.MEAN,
            trans_reduce=valid.EValidECDFReduce.MEAN,
        ),
    ),
    (
        "median_curves",
        valid.ValidECDFView(
            nuis_reduce=valid.EValidECDFReduce.MEDIAN,
            trans_reduce=valid.EValidECDFReduce.MEDIAN,
        ),
        valid.ValidECDFView(
            nuis_reduce=valid.EValidECDFReduce.MEDIAN,
            trans_reduce=valid.EValidECDFReduce.MEDIAN,
        ),
    ),
    (
        "selected_scenarios",
        valid.ValidECDFView(bounds=False, scenario_idxs=((0, 0), (2, 2))),
        valid.ValidECDFView(bounds=False, scenario_idxs=((0, 1), (1, 2))),
    ),
    (
        "independent_views",
        valid.ValidECDFView(nuis_reduce=valid.EValidECDFReduce.MEDIAN),
        valid.ValidECDFView(
            nuis_reduce=valid.EValidECDFReduce.MEAN,
            trans_reduce=valid.EValidECDFReduce.MEDIAN,
        ),
    ),
)
for name, sim_view, exp_view in views:
    ecdf_plot: validplot.ValidECDFPlot = validplot.plot_sim_exp_cdfs(
        valid_case,
        observable_tag,
        point_idxs=None,
        sim_view=sim_view,
        exp_view=exp_view,
    )
    ecdf_plot.figure.savefig(output_path / f"{name}.png", bbox_inches="tight")
    if not SHOW_PLOTS:
        plt.close(ecdf_plot.figure)
sim_plot: validplot.ValidECDFPlot = validplot.plot_sim_ecdfs(
    valid_case,
    observable_tag,
    point_idxs=None,
    view=valid.ValidECDFView(nuis_reduce=valid.EValidECDFReduce.MEDIAN),
)
sim_plot.figure.savefig(output_path / "sim_median_nuisance.png", bbox_inches="tight")
exp_plot: validplot.ValidECDFPlot = validplot.plot_exp_ecdfs(
    valid_case, observable_tag, point_idxs=None
)
exp_plot.figure.savefig(output_path / "exp_envelopes.png", bbox_inches="tight")

# %%
# 4. Normalisation uses one reference constant per panel: the mean sim quantile
# at p=.5 or p=.9. The physical axis remains below; the dimensionless axis above
# is x/reference, not a probability-dependent displacement of each curve.
for probability in (0.5, 0.9):
    ecdf_plot = validplot.plot_sim_exp_cdfs(valid_case, observable_tag, point_idxs=None)
    validplot.add_normalised_ecdf_axis(
        ecdf_plot, validplot.ValidECDFReference(probability=probability)
    )
    validplot.add_metric_overlay(ecdf_plot, valid.MetricAVM())
    ecdf_plot.figure.savefig(
        output_path / f"avm_normalised_p{probability:g}.png", bbox_inches="tight"
    )
    if not SHOW_PLOTS:
        plt.close(ecdf_plot.figure)
ks_plot: validplot.ValidECDFPlot = validplot.plot_sim_exp_cdfs(
    valid_case, observable_tag, point_idxs=None
)
validplot.add_metric_overlay(ks_plot, valid.MetricKS())
ks_plot.figure.savefig(output_path / "ks_bound_distances.png", bbox_inches="tight")

# Support shifts search original conditional comparisons. Integration areas
# require selecting an actual sim/exp pair; summaries are not pseudo-observations.
for metric_output, direction in (
    (valid.EMetricOutput.D_PLUS, "d_plus"),
    (valid.EMetricOutput.D_MINUS, "d_minus"),
):
    support_plot: validplot.ValidECDFPlot = validplot.plot_sim_exp_cdfs(
        valid_case, observable_tag, point_idxs=None
    )
    validplot.add_metric_overlay(support_plot, valid.MetricMAVM(), output=metric_output)
    support_plot.figure.savefig(
        output_path / f"mavm_{direction}_support.png", bbox_inches="tight"
    )
    area_plot: validplot.ValidECDFPlot = validplot.plot_sim_exp_cdfs(
        valid_case,
        observable_tag,
        point_idxs=None,
        sim_view=valid.ValidECDFView(trans_idxs=(1,), nuis_idxs=(1,)),
        exp_view=valid.ValidECDFView(trans_idxs=(0,), nuis_idxs=(1,)),
    )
    validplot.add_metric_overlay(
        area_plot,
        valid.MetricMAVM(),
        output=metric_output,
        display=validplot.EValidMetricDisplay.INTEGRATION_AREA,
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
