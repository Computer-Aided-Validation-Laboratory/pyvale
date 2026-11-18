# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Transient nuisance extrema, shifted supports and coherent case ranking.

Both sources have transferable and nuisance axes. The exp bias increases with
position and time; nuisance cases have different spreads. Changes of controlling
case create piecewise trends, not random fields. KS/CvM can saturate even while
physical-distance metrics continue growing. No observations are randomly drawn.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

import pyvale.validation as vl

matplotlib.use("Agg")
SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")
output_path = Path.cwd() / "pyvale-output" / "valid_ex6_nuisance_profiles"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Keep all six axes distinct. Equal repeat counts allow paired metrics too.
coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
times = np.array([0.0, 1.0, 2.0])
base = 10.0 + 2.0 * coords[:, :1] + times
bias = 1.0 + 2.0 * coords[:, :1] + 0.5 * times
sim_shift = np.array([[0.0, 1.0, 2.0], [2.0, 4.0, 8.0]])
exp_shift = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 7.0]])
quantile_grid = np.array([-1.0, -1 / 3, 1 / 3, 1.0])
sim_samples = (
    base[:, None, :, None, None, None]
    + sim_shift[None, None, None, :, :, None]
    + np.array([0.2, 1.0, 2.0])[None, None, None, None, :, None] * quantile_grid
)
exp_samples = (
    (base + bias)[:, None, :, None, None, None]
    + exp_shift[None, None, None, :, :, None]
    + np.array([0.4, 0.8, 1.2])[None, None, None, None, :, None] * quantile_grid
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

# %%
# 2. Value profiling answers "which nuisance samples minimise/maximise the
# metric?" Transferable axes remain. Extrema over all epistemic axes answer a
# different question and reduce those axes too. Indices always identify original
# comparisons; they can differ between outputs and physical cells.
summary: dict[str, object] = {}
for metric_type, field in valid_result.metrics[observable_tag].items():
    profile: vl.ValidMetricExtrema = vl.profile_epis_nuisance_extrema(
        valid_result, observable_tag, metric_type
    )
    all_extrema: vl.ValidMetricExtrema = vl.find_metric_epis_extrema(
        valid_result, observable_tag, metric_type
    )
    summary[metric_type.name] = {
        "retained_axes": [axis.value for axis in profile.axes],
        "all_epistemic_axes": [axis.value for axis in all_extrema.axes],
    }
    # Reuse generic metric plotting with profiled fields. The profiles do not
    # calculate metrics again; both fields borrow their already reduced arrays.
    maximum_result = vl.ValidResult(
        observables,
        {
            observable_tag: {
                metric_type: vl.MetricField(profile.axes, profile.maximum)
            }
        },
        valid_result.metric_configs,
    )
    minimum_result = vl.ValidResult(
        observables,
        {
            observable_tag: {
                metric_type: vl.MetricField(profile.axes, profile.minimum)
            }
        },
        valid_result.metric_configs,
    )
    for output in field.outputs:
        name = f"{metric_type.name.lower()}_{output.name.lower()}"
        fig: Figure
        plot_axes: np.ndarray
        fig, plot_axes = vl.plot_metric_time_points(
            maximum_result,
            observable_tag,
            metric_type,
            point_idxs=(0, 1),
            selectors={
                vl.EValidResAxis.SIM_TRANS: 0,
                vl.EValidResAxis.EXP_TRANS: 0,
            },
            output=output,
        )
        for point, ax in enumerate(plot_axes.flat):
            heading = ax.get_title()
            ax.lines[0].set_label("Maximum over nuisance")
            vl.plot_valid_metric_time(
                minimum_result,
                observable_tag,
                metric_type,
                selectors={
                    vl.EValidResAxis.POINT: point,
                    vl.EValidResAxis.SIM_TRANS: 0,
                    vl.EValidResAxis.EXP_TRANS: 0,
                },
                output=output,
                ax=ax,
            )
            ax.set_title(heading)
            ax.lines[-1].set_label("Minimum over nuisance")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3))
        fig.savefig(output_path / f"{name}_profile.png", bbox_inches="tight")
        np.savetxt(
            output_path / f"{name}_profile.csv",
            np.column_stack(
                (
                    times,
                    profile.minimum[output][0, 0, :, 0, 0],
                    profile.maximum[output][0, 0, :, 0, 0],
                    profile.minimum_indices[output][0, 0, :, 0, 0],
                    profile.maximum_indices[output][0, 0, :, 0, 0],
                )
            ),
            delimiter=",",
            header="time,minimum,maximum,min_sim_nuis,min_exp_nuis,max_sim_nuis,max_exp_nuis",
            comments="",
        )
        if not SHOW_PLOTS:
            plt.close(fig)

# %%
# 3. Support profiling asks which ECDF +/- its own d reaches furthest. This is
# not a globally largest d added to the sim envelope. Origins may change along p.
supports: vl.ValidSupportBounds = vl.profile_epis_nuisance_supports(
    valid_case, valid_result, observable_tag, vl.EMetricType.MAVM
)
summary["support_index_axes"] = [axis.value for axis in supports.index_axes]
summary["upper_origin_at_point0_time2"] = supports.upper_indices[0, 0, 2, 0, 0].tolist()
for metric_output, direction in (
    (vl.EMetricOut.D_PLUS, "d_plus"),
    (vl.EMetricOut.D_MINUS, "d_minus"),
):
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
        output=metric_output,
        selectors={vl.EValidResAxis.SIM_TRANS: 0, vl.EValidResAxis.EXP_TRANS: 0},
    )
    ecdf_plot.figure.savefig(
        output_path / f"mavm_{direction}_nuisance_support.png", bbox_inches="tight"
    )
    if not SHOW_PLOTS:
        plt.close(ecdf_plot.figure)
avm_supports: vl.ValidSupportBounds = vl.find_metric_epis_supports(
    valid_case, valid_result, observable_tag, vl.EMetricType.AVM
)
ecdf_plot = vl.plot_sim_exp_cdfs(
    valid_case, observable_tag, point_idxs=None, time_idx=2
)
vl.add_support_overlay(
    ecdf_plot, avm_supports, label="AVM symmetric magnitude envelope"
)
ecdf_plot.figure.savefig(output_path / "avm_original_support.png", bbox_inches="tight")
if not SHOW_PLOTS:
    plt.close(ecdf_plot.figure)

# %%
# 4. Rank whole nuisance cases: first envelope transferable uncertainty while
# retaining nuisance axes. Total width and additional outward extension answer
# different questions. A median-ranked case is actual data, unlike ex05's
# synthetic pointwise nuisance median. Stable ties retain canonical index order.
by_case: vl.ValidSupportBounds = vl.find_metric_epis_supports(
    valid_case,
    valid_result,
    observable_tag,
    vl.EMetricType.MAVM,
    reduce_axes=(vl.EValidResAxis.SIM_TRANS, vl.EValidResAxis.EXP_TRANS),
)
for score in vl.EValidNuisanceScore:
    ranking: vl.ValidNuisanceRanking = vl.rank_epis_nuisance_cases(
        by_case, score=score
    )
    score_name = score.name.lower()
    valid_count = int(ranking.valid_counts[0, 0, 2])
    order = ranking.order[0, 0, 2]
    selected_rows = (
        int(order[0]),
        int(order[(valid_count - 1) // 2]),
        int(order[valid_count - 1]),
    )
    summary[score_name] = {
        "cases": ranking.cases.tolist(),
        "scores": ranking.scores[0, 0, 2].tolist(),
        "order": order.tolist(),
        "selected_rows": selected_rows,
    }
    for rank_name, row in zip(
        ("minimum", "median", "maximum"), selected_rows, strict=True
    ):
        sim_nuis, exp_nuis = (int(index) for index in ranking.cases[row])
        for metric_output, direction in (
            (vl.EMetricOut.D_PLUS, "d_plus"),
            (vl.EMetricOut.D_MINUS, "d_minus"),
        ):
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
                output=metric_output,
                selectors={
                    vl.EValidResAxis.SIM_NUIS: sim_nuis,
                    vl.EValidResAxis.EXP_NUIS: exp_nuis,
                },
                label=f"{rank_name} {score_name}: nuisance ({sim_nuis}, {exp_nuis})",
            )
            ecdf_plot.figure.savefig(
                output_path / f"{score_name}_{rank_name}_{direction}.png",
                bbox_inches="tight",
            )
            if not SHOW_PLOTS:
                plt.close(ecdf_plot.figure)
(output_path / "profiling.json").write_text(
    json.dumps(summary, indent=2, allow_nan=False)
)
if SHOW_PLOTS:
    plt.show()
