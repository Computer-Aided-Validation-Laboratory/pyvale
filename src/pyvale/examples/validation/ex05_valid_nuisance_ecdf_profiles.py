# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Nuisance ECDF profiles
================================================================================

Profile nuisance uncertainty in temperature ECDFs and compare the effect on KS
and MAVM.
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

output_path = Path.cwd() / "pyvale-output" / "valid_ex05_nuisance_ecdf_profiles"
output_path.mkdir(parents=True, exist_ok=True)
fig_nums = count(1)

# %%
# 1. Separate the sources of variation
# ------------------------------------
#
# Transferable cases represent conditions of interest. Nuisance cases capture
# uncertain conditions over which you may want to summarise. Aleatory samples
# represent repeat variability within a joint case. Broadcasting builds arrays
# ordered by point, component, transferable case, nuisance case, and aleatory
# sample.
coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

ground_truth = np.array([100.0, 120.0])  # degC at the two sites

sim_epistemic_trans_shift = np.array([2.0, 4.0, 8.0])
sim_epistemic_nuis_shift = np.array([0.0, 178.0, 8.0])
exp_epistemic_trans_shift = np.array([0.0, 6.0])
exp_epistemic_nuis_shift = np.array([0.0, 2.0, 12.0])

sim_aleatory_deviations = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
exp_aleatory_deviations = np.array([-8.0, -2.0, 2.0, 8.0])

sim_samples = (
    ground_truth[:, None, None, None, None]
    + sim_epistemic_trans_shift[None, None, :, None, None]
    + sim_epistemic_nuis_shift[None, None, None, :, None]
    + sim_aleatory_deviations
)

exp_samples = (
    ground_truth[:, None, None, None, None]
    + exp_epistemic_trans_shift[None, None, :, None, None]
    + exp_epistemic_nuis_shift[None, None, None, :, None]
    + exp_aleatory_deviations
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

# %%
# 2. Compare the envelope and nuisance profiles
# ---------------------------------------------
#
# The outer envelope spans all original cases. Profiling takes a pointwise
# quantile summary across nuisance cases at each transferable case, then
# envelopes the resulting curves. The large simulation nuisance shift makes
# the mean, median, and lower quartile behave differently.
for nn, pp in (
    ("sim", vl.plot_sim_ecdfs),
    ("exp", vl.plot_exp_ecdfs),
    ("sim_exp", vl.plot_sim_exp_cdfs),
):
    ecdf_plot = pp(
        valid_case,
        observable_tag,
        point_idxs=None,
        nuisance_profile=vl.EValidECDFReduce.MEDIAN,
    )
    ecdf_plot.figure.suptitle("All-case envelope vs. median nuisance profile")
    fig_file = (
        f"val_ex05_fig{next(fig_nums):02d}_"
        f"ecdf_{nn}_nuis_envelope_vs_prof_median.png"
    )
    ecdf_plot.figure.savefig(output_path / fig_file, bbox_inches="tight")

for nn, tt, rr, qq in (
    ("mean", "mean", vl.EValidECDFReduce.MEAN, None),
    ("q25", "25th-percentile", vl.EValidECDFReduce.QUANTILE, 0.25),
):
    ecdf_plot = vl.plot_sim_exp_cdfs(
        valid_case,
        observable_tag,
        point_idxs=None,
        nuisance_profile=rr,
        nuisance_profile_quantile=qq,
    )
    ecdf_plot.figure.suptitle(f"All-case envelope vs. {tt} nuisance profile")
    fig_file = (
        f"val_ex05_fig{next(fig_nums):02d}_"
        f"ecdf_sim_exp_nuis_envelope_vs_prof_{nn}.png"
    )
    ecdf_plot.figure.savefig(output_path / fig_file, bbox_inches="tight")

# %%
# 3. Compare KS gaps
# ------------------
#
# Use the same median reduction on both sources. The KS overlay measures the
# largest vertical gap between the selected ECDF bounds.
median_view = vl.ValidECDFView(nuis_reduce=vl.EValidECDFReduce.MEDIAN)
for nn, tt, vv in (
    ("envelope", "all-case envelope", None),
    ("prof_median", "median nuisance profile", median_view),
):
    ks_plot = vl.plot_sim_exp_cdfs(
        valid_case, observable_tag, point_idxs=None, sim_view=vv, exp_view=vv
    )
    vl.add_metric_overlay(ks_plot, vl.MetricKS())
    ks_plot.figure.suptitle(f"KS gap between {tt} bounds")
    fig_file = f"val_ex05_fig{next(fig_nums):02d}_ks_nuis_{nn}_gap.png"
    ks_plot.figure.savefig(output_path / fig_file, bbox_inches="tight")

# %%
# 4. Compare MAVM support envelopes
# ---------------------------------
#
# The first overlay spans the shifted supports across original case pairs. For
# the median view, calculate support from the profiled ECDF bounds at each
# point and add it to the matching panel.
mavm_plot = vl.plot_sim_exp_cdfs(valid_case, observable_tag, point_idxs=None)
vl.add_metric_overlay(mavm_plot, vl.MetricMAVM())
mavm_plot.figure.suptitle("MAVM support envelope over original scenario pairs")
fig_file = f"val_ex05_fig{next(fig_nums):02d}_mavm_support_envelope_original_pairs.png"
mavm_plot.figure.savefig(output_path / fig_file, bbox_inches="tight")

fig, plot_axes = vl.create_valid_plot_grid(len(coords))
for ii, aa in enumerate(plot_axes.flat):
    sim_bounds = vl.find_ecdf_epis_bounds(
        valid_case.sim[observable_tag], point_idx=ii, nuis_reduce=median_view.nuis_reduce
    )
    exp_bounds = vl.find_ecdf_epis_bounds(
        valid_case.exp[observable_tag], point_idx=ii, nuis_reduce=median_view.nuis_reduce
    )
    median_mavm_plot = vl.plot_sim_exp_cdfs(
        valid_case,
        observable_tag,
        point_idxs=(ii,),
        sim_view=median_view,
        exp_view=median_view,
        ax=aa,
    )
    vl.add_support_overlay(
        median_mavm_plot, vl.calc_mavm_over_epis_bounds(sim_bounds, exp_bounds)
    )

fig.suptitle("MAVM support envelope between median nuisance profile bounds")
fig_file = (
    f"val_ex05_fig{next(fig_nums):02d}_"
    "mavm_support_envelope_nuis_prof_median.png"
)
fig.savefig(output_path / fig_file, bbox_inches="tight")

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
