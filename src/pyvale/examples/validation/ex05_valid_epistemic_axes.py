# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Epistemic ECDF views
================================================================================

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
# A large sim nuisance shift exposes the sensitivity of the mean.

coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

ground_truth = np.array([100.0, 120.0])  # degC at the two sites

# Each scenario combines one transferable shift and one nuisance shift in degC.
sim_epistemic_trans_shift = np.array([2.0, 4.0, 8.0])
sim_epistemic_nuis_shift = np.array([0.0, 178.0, 8.0])
exp_epistemic_trans_shift = np.array([0.0, 6.0])
exp_epistemic_nuis_shift = np.array([0.0, 2.0, 12.0])

# Fixed deviations provide aleatory spread within every scenario.
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
# 2. Compare the all-case envelope with a nuisance profile. Each profile
# reduces nuisance quantiles within each transferable case, then envelopes
# the resulting curves over the transferable cases.
for source_name, plot_sources in (
    ("sim", vl.plot_sim_ecdfs),
    ("exp", vl.plot_exp_ecdfs),
    ("sim_exp", vl.plot_sim_exp_cdfs),
):
    ecdf_plot = plot_sources(
        valid_case,
        observable_tag,
        point_idxs=None,
        nuisance_profile=vl.EValidECDFReduce.MEDIAN,
    )
    ecdf_plot.figure.suptitle("All-case envelope vs. median nuisance profile")
    ecdf_plot.figure.savefig(
        output_path / f"ecdf_{source_name}_nuis_envelope_vs_prof_median.png",
        bbox_inches="tight",
    )

for name, title, reduction, probability in (
    ("mean", "mean", vl.EValidECDFReduce.MEAN, None),
    ("q25", "25th-percentile", vl.EValidECDFReduce.QUANTILE, 0.25),
):
    ecdf_plot = vl.plot_sim_exp_cdfs(
        valid_case,
        observable_tag,
        point_idxs=None,
        nuisance_profile=reduction,
        nuisance_profile_quantile=probability,
    )
    ecdf_plot.figure.suptitle(f"All-case envelope vs. {title} nuisance profile")
    ecdf_plot.figure.savefig(
        output_path / f"ecdf_sim_exp_nuis_envelope_vs_prof_{name}.png",
        bbox_inches="tight",
    )

# %%
# 3. KS measures the vertical gap between corresponding displayed bounds.
median_view = vl.ValidECDFView(nuis_reduce=vl.EValidECDFReduce.MEDIAN)
for name, title, view in (
    ("envelope", "all-case envelope", None),
    ("prof_median", "median nuisance profile", median_view),
):
    ks_plot = vl.plot_sim_exp_cdfs(
        valid_case, observable_tag, point_idxs=None, sim_view=view, exp_view=view
    )
    vl.add_metric_overlay(ks_plot, vl.MetricKS())
    ks_plot.figure.suptitle(f"KS gap between {title} bounds")
    ks_plot.figure.savefig(output_path / f"ks_nuis_{name}_gap.png", bbox_inches="tight")

# %%
# 4. MAVM shifts the simulation support using original scenario comparisons.
# The shaded envelope spans both directional shifts.
mavm_plot = vl.plot_sim_exp_cdfs(valid_case, observable_tag, point_idxs=None)
vl.add_metric_overlay(mavm_plot, vl.MetricMAVM())
mavm_plot.figure.suptitle("MAVM support envelope over original scenario pairs")
mavm_plot.figure.savefig(
    output_path / "mavm_support_envelope_original_pairs.png",
    bbox_inches="tight",
)

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
