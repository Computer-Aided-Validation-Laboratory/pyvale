# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Nuisance metric profiles
================================================================================
"""

from itertools import count
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyvale.validation as vl

matplotlib.use("Agg")

SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")

output_path = Path.cwd() / "pyvale-output" / "valid_ex06_nuisance_metric_profiles"
output_path.mkdir(parents=True, exist_ok=True)
figure_numbers = count(1)

# %%
coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
times = np.array([0.0, 1.0, 2.0])
ground_truth = np.array([100.0, 120.0])[:, None] + 10.0 * times
exp_bias = np.array([10.0, 30.0])[:, None] + 5.0 * times

sim_epistemic_trans_shift = np.array([2.0, 4.0, 8.0])
sim_epistemic_nuis_shift = np.array([0.0, 178.0, 8.0])
exp_epistemic_trans_shift = np.array([0.0, 6.0])
exp_epistemic_nuis_shift = np.array([0.0, 2.0, 12.0])

aleatory_grid = np.array([-1.0, -1 / 3, 1 / 3, 1.0])
sim_aleatory_half_width = np.array([2.0, 10.0, 20.0])
exp_aleatory_half_width = np.array([4.0, 8.0, 12.0])
sim_aleatory_deviations = sim_aleatory_half_width[:, None] * aleatory_grid
exp_aleatory_deviations = exp_aleatory_half_width[:, None] * aleatory_grid

sim_samples = (
    ground_truth[:, None, :, None, None, None]
    + sim_epistemic_trans_shift[None, None, None, :, None, None]
    + sim_epistemic_nuis_shift[None, None, None, None, :, None]
    + sim_aleatory_deviations
)
exp_samples = (
    (ground_truth + exp_bias)[:, None, :, None, None, None]
    + exp_epistemic_trans_shift[None, None, None, :, None, None]
    + exp_epistemic_nuis_shift[None, None, None, None, :, None]
    + exp_aleatory_deviations
)
axes = (
    vl.EValidAxis.POINT,
    vl.EValidAxis.COMPONENT,
    vl.EValidAxis.TIME,
    vl.EValidAxis.EPISTEMIC_TRANS,
    vl.EValidAxis.EPISTEMIC_NUIS,
    vl.EValidAxis.ALEATORY,
)
observable_tag = "temperature"
observables = {
    observable_tag: vl.ValidObservable(
        components=("temperature",),
        geometry=vl.EValidGeometry.POINT,
        coords=coords,
        times=times,
        phys_units="degC",
        coord_units="m",
        time_units="s",
    )
}
valid_case = vl.ValidCase(
    observables,
    {observable_tag: vl.ValidArray(sim_samples, axes)},
    {observable_tag: vl.ValidArray(exp_samples, axes)},
)
case_path = output_path / "validation_case.npz"
vl.save_valid_case(case_path, valid_case)
valid_case = vl.load_valid_case(case_path)

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

result_path = output_path / "validation_result.npz"
vl.save_valid_result(result_path, valid_result)
valid_result = vl.load_valid_result(result_path)

# %%
for mm, ff in valid_result.metrics[observable_tag].items():
    profile: vl.ValidMetricExtrema = vl.profile_epis_nuisance_extrema(
        valid_result, observable_tag, mm
    )
    fig, _ = vl.plot_metric_extrema_time(
        valid_result,
        observable_tag,
        mm,
        profile=profile,
        selectors={
            vl.EValidResAxis.SIM_TRANS: 0,
            vl.EValidResAxis.EXP_TRANS: 0,
        },
    )
    profile_name = mm.name.lower()
    if len(ff.outputs) == 1:
        profile_name += f"_{next(iter(ff.outputs)).name.lower()}"
    fig.savefig(
        output_path
        / (
            f"val_ex06_fig{next(figure_numbers):02d}_"
            f"{profile_name}_profile.png"
        ),
        bbox_inches="tight",
    )
    if not SHOW_PLOTS:
        plt.close(fig)
    vl.save_metric_extrema(output_path / f"{profile_name}_profile.npz", profile)

mavm_profile = vl.load_metric_extrema(output_path / "mavm_profile.npz")
fig, _ = vl.plot_metric_extrema_ecdfs(
    valid_case,
    observable_tag,
    vl.EMetricType.MAVM,
    profile=mavm_profile,
    output=vl.EMetricOut.D_PLUS,
    selectors={
        vl.EValidResAxis.POINT: 0,
        vl.EValidResAxis.TIME: 2,
        vl.EValidResAxis.SIM_TRANS: 0,
        vl.EValidResAxis.EXP_TRANS: 0,
    },
)
fig.savefig(
    output_path
    / (
        f"val_ex06_fig{next(figure_numbers):02d}_"
        f"mavm_d_plus_extrema_ecdfs.png"
    ),
    bbox_inches="tight",
)
if not SHOW_PLOTS:
    plt.close(fig)

# %%
supports: vl.ECDFSuppBounds = vl.profile_epis_nuisance_supports(
    valid_case, valid_result, observable_tag, vl.EMetricType.MAVM
)
ecdf_plot: vl.ECDFPlot = vl.plot_sim_exp_cdfs(
    valid_case,
    observable_tag,
    point_idxs=None,
    time_idx=2,
    sim_view=vl.ValidECDFView(trans_idxs=(0,)),
    exp_view=vl.ValidECDFView(trans_idxs=(0,)),
)
vl.add_support_overlay(
    ecdf_plot,
    supports,
    selectors={vl.EValidResAxis.SIM_TRANS: 0, vl.EValidResAxis.EXP_TRANS: 0},
)
ecdf_plot.figure.savefig(
    output_path
    / (
        f"val_ex06_fig{next(figure_numbers):02d}_"
        f"mavm_nuisance_support_envelope.png"
    ),
    bbox_inches="tight",
)
if not SHOW_PLOTS:
    plt.close(ecdf_plot.figure)
avm_supports: vl.ECDFSuppBounds = vl.find_metric_epis_supports(
    valid_case, valid_result, observable_tag, vl.EMetricType.AVM
)
ecdf_plot = vl.plot_sim_exp_cdfs(
    valid_case, observable_tag, point_idxs=None, time_idx=2
)
vl.add_support_overlay(
    ecdf_plot, avm_supports
)
ecdf_plot.figure.savefig(
    output_path / f"val_ex06_fig{next(figure_numbers):02d}_avm_original_support.png",
    bbox_inches="tight",
)
if not SHOW_PLOTS:
    plt.close(ecdf_plot.figure)

# %%
by_case: vl.ECDFSuppBounds = vl.find_metric_epis_supports(
    valid_case,
    valid_result,
    observable_tag,
    vl.EMetricType.MAVM,
    reduce_axes=(vl.EValidResAxis.SIM_TRANS, vl.EValidResAxis.EXP_TRANS),
)
for ss in vl.EValidNuisanceScore:
    ranking: vl.ECDFNuisanceRank = vl.rank_epis_nuisance_cases(
        by_case, score=ss
    )
    score_name = ss.name.lower()
    valid_count = int(ranking.valid_counts[0, 0, 2])
    order = ranking.order[0, 0, 2]
    selected_rows = (
        int(order[0]),
        int(order[(valid_count - 1) // 2]),
        int(order[valid_count - 1]),
    )
    for nn, rr in zip(
        ("minimum", "median", "maximum"), selected_rows, strict=True
    ):
        sim_nuis, exp_nuis = (int(ii) for ii in ranking.cases[rr])
        ecdf_plot = vl.plot_sim_exp_cdfs(
            valid_case,
            observable_tag,
            time_idx=2,
            sim_view=vl.ValidECDFView(nuis_idxs=(sim_nuis,)),
            exp_view=vl.ValidECDFView(nuis_idxs=(exp_nuis,)),
        )
        vl.add_support_overlay(
            ecdf_plot,
            by_case,
            selectors={
                vl.EValidResAxis.SIM_NUIS: sim_nuis,
                vl.EValidResAxis.EXP_NUIS: exp_nuis,
            },
        )
        ecdf_plot.figure.savefig(
            output_path
            / (
                f"val_ex06_fig{next(figure_numbers):02d}_"
                f"{score_name}_{nn}_support_envelope.png"
            ),
            bbox_inches="tight",
        )
        if not SHOW_PLOTS:
            plt.close(ecdf_plot.figure)
if SHOW_PLOTS:
    plt.show()
