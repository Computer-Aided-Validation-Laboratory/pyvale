# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Explore transferable/nuisance ECDF views and composable metric overlays.

Analytic sample arrays keep this example independent of downloaded datasets.
Both sources have transferable and nuisance axes, with different numbers of
transferable and aleatory samples. Nothing pools the conditional observations.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from pyvale import valid
from pyvale.valid import validplot

matplotlib.use("Agg")
SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")
output_path = Path.cwd() / "pyvale-output" / "valid_ex11_epistemic_axes"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Build two physical sites, three times and conditional aleatory samples.
# Sim has (3 transferable, 3 nuisance, 5 aleatory); exp has (2, 3, 4).
# Nuisance shifts are deliberately asymmetric so mean and median differ.
observable_tag = "temperature"
times = np.array([0.0, 1.0, 2.0])
coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
physical_baseline = 10.0 + 2.0 * np.arange(2)[:, None, None, None, None, None]
physical_baseline = physical_baseline + times[None, None, :, None, None, None]
sim_shifts = 0.2 * np.array([[1.0, 9.0, 8.0], [2.0, 3.0, 7.0], [4.0, 6.0, 5.0]])
exp_shifts = 0.2 * np.array([[0.0, 1.0, 7.0], [3.0, 4.0, 8.0]])
sim_samples = (
    physical_baseline
    + sim_shifts[None, None, None, :, :, None]
    + np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
)
exp_samples = (
    physical_baseline
    + exp_shifts[None, None, None, :, :, None]
    + np.array([-0.8, -0.2, 0.2, 0.8])
)
axes = (
    valid.EValidAxis.POINT,
    valid.EValidAxis.COMPONENT,
    valid.EValidAxis.TIME,
    valid.EValidAxis.EPISTEMIC_TRANS,
    valid.EValidAxis.EPISTEMIC_NUIS,
    valid.EValidAxis.ALEATORY,
)
valid_case = valid.ValidCase(
    {
        observable_tag: valid.ValidObservable(
            components=("temperature",),
            geometry=valid.EValidGeometry.POINT,
            coords=coords,
            times=times,
            phys_units="degC",
            coord_units="m",
            time_units="s",
        )
    },
    {observable_tag: valid.ValidArray(sim_samples, axes)},
    {
        observable_tag: valid.ValidArray(
            exp_samples, axes, valid.EpistemicOffsets(0.15, 0.25)
        )
    },
)
valid_result: valid.ValidResult = valid.ValidEngine(
    [valid.MetricAVM(), valid.MetricKS(), valid.MetricMAVM()]
).run(valid_case)

# %%
# 2. Compare outer bounds, nuisance reductions, central curves and selected pairs.
# KEEP retains an axis; nuisance reduction precedes transferable reduction.
# A mean/median quantile curve is a synthetic summary, not a pooled ECDF.
views = (
    ("outer_outer", valid.ValidECDFView(), valid.ValidECDFView()),
    (
        "median_nuisance",
        valid.ValidECDFView(nuis_reduce=valid.EValidECDFReduce.MEDIAN),
        valid.ValidECDFView(nuis_reduce=valid.EValidECDFReduce.MEDIAN),
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
        valid.ValidECDFView(scenario_idxs=((0, 0), (2, 2))),
        valid.ValidECDFView(scenario_idxs=((0, 1), (1, 2))),
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
        time_idx=2,
        sim_view=sim_view,
        exp_view=exp_view,
    )
    ecdf_plot.figure.savefig(output_path / f"{name}.png", bbox_inches="tight")
    if not SHOW_PLOTS:
        plt.close(ecdf_plot.figure)

# Separate source plots use the same view rules and automatically retain labels.
sim_plot: validplot.ValidECDFPlot = validplot.plot_sim_ecdfs(
    valid_case,
    observable_tag,
    point_idxs=None,
    time_idx=2,
    view=valid.ValidECDFView(nuis_reduce=valid.EValidECDFReduce.MEDIAN),
)
sim_plot.figure.savefig(output_path / "sim_median_nuisance.png", bbox_inches="tight")
# Offsets are nominal by default. None explicitly includes all coherent shifts.
exp_plot: validplot.ValidECDFPlot = validplot.plot_exp_ecdfs(
    valid_case, observable_tag, point_idxs=None, time_idx=2, offset=None
)
exp_plot.figure.savefig(output_path / "exp_offsets.png", bbox_inches="tight")

# %%
# 3. Compose a metric overlay and a constant normalised secondary horizontal axis.
# Each panel uses its own mean sim quantile at p=.5 as one scaling constant.
# This does not subtract a different reference at every cumulative probability.
ecdf_plot = validplot.plot_sim_exp_cdfs(
    valid_case, observable_tag, point_idxs=None, time_idx=2
)
validplot.add_normalised_ecdf_axis(
    ecdf_plot, validplot.ValidECDFReference(probability=0.5)
)
validplot.add_metric_overlay(ecdf_plot, valid.MetricAVM())
ecdf_plot.figure.savefig(
    output_path / "avm_normalised_support.png", bbox_inches="tight"
)
ks_plot: validplot.ValidECDFPlot = validplot.plot_sim_exp_cdfs(
    valid_case, observable_tag, point_idxs=None, time_idx=2
)
validplot.add_metric_overlay(ks_plot, valid.MetricKS())
ks_plot.figure.savefig(output_path / "ks_bound_distances.png", bbox_inches="tight")

# %%
# 4. Show d+ and d- separately, including both supported overlay representations.
# Shift envelopes search all original selected comparisons. A large distance on
# an interior sim ECDF does not necessarily control the outer shifted support.
for metric_output, direction in (
    (valid.EMetricOutput.D_PLUS, "d_plus"),
    (valid.EMetricOutput.D_MINUS, "d_minus"),
):
    support_plot: validplot.ValidECDFPlot = validplot.plot_sim_exp_cdfs(
        valid_case, observable_tag, point_idxs=None, time_idx=2
    )
    validplot.add_metric_overlay(support_plot, valid.MetricMAVM(), output=metric_output)
    support_plot.figure.savefig(
        output_path / f"mavm_{direction}_support.png", bbox_inches="tight"
    )
    # Integration areas belong to an original conditional pair, not synthetic
    # envelope boundaries. Select both views explicitly to display those curves.
    area_plot: validplot.ValidECDFPlot = validplot.plot_sim_exp_cdfs(
        valid_case,
        observable_tag,
        point_idxs=None,
        time_idx=2,
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
    # Numerical results retain all epistemic axes. A metric time plot still
    # needs an explicit original scenario selection on each non-singleton axis.
    fig: Figure
    metric_axes: np.ndarray
    fig, metric_axes = validplot.plot_valid_metric_time_points(
        valid_result,
        observable_tag,
        valid.EMetricType.MAVM,
        point_idxs=(0, 1),
        selectors={
            valid.EValidResAxis.SIM_TRANS: 1,
            valid.EValidResAxis.SIM_NUIS: 1,
            valid.EValidResAxis.EXP_TRANS: 0,
            valid.EValidResAxis.EXP_NUIS: 1,
            valid.EValidResAxis.EXP_OFFSET: valid.EEpisOffset.NOMINAL,
        },
        output=metric_output,
    )
    fig.savefig(output_path / f"mavm_{direction}_time.png", bbox_inches="tight")
    if not SHOW_PLOTS:
        plt.close(support_plot.figure)
        plt.close(area_plot.figure)
        plt.close(fig)

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
