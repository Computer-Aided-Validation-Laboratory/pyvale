# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Transient coherent offsets, experimental confidence and epistemic envelopes.

The imposed exp bias is .4+.5t degC. Calibration offsets shift an entire
conditional distribution, whereas MAVM confidence depends on its observations.
These are distinct from the epistemic scenario envelope shown in ex04-ex06.
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
output_path = Path.cwd() / "pyvale-output" / "valid_ex7_offsets"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Two points and three times share a deterministic aleatory sampling grid.
coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
times = np.array([0.0, 1.0, 2.0])
baseline = 10.0 + 2.0 * coords[:, :1] + times
sim_samples = baseline[:, None, :, None] + np.array([-0.6, -0.2, 0.2, 0.6])
bias = 0.4 + 0.5 * times
exp_samples = sim_samples + bias[None, None, :, None]
lower_offset = 0.15 + 0.1 * coords[:, :1] + 0.05 * times
upper_offset = 0.4 + 0.2 * coords[:, :1] + 0.1 * times
axes = (
    vl.EValidAxis.POINT,
    vl.EValidAxis.COMPONENT,
    vl.EValidAxis.TIME,
    vl.EValidAxis.ALEATORY,
)
observable_tag = "temperature"
valid_case = vl.ValidCase(
    {
        observable_tag: vl.ValidObservable(
            components=("temperature",),
            geometry=vl.EValidGeometry.POINT,
            coords=coords,
            times=times,
            phys_units="degC",
            coord_units="m",
            time_units="s",
        )
    },
    {observable_tag: vl.ValidArray(sim_samples, axes)},
    {
        observable_tag: vl.ValidArray(
            exp_samples,
            axes,
            vl.EpistemicOffsets(
                lower_offset,
                upper_offset,
                (vl.EValidAxis.POINT, vl.EValidAxis.TIME),
            ),
        )
    },
)
valid_result: vl.ValidResult = vl.ValidEngine(
    [vl.MetricAVM(), vl.MetricMAVM()]
).run(valid_case)

# %%
# 2. Only the explicit exp_offset=None expands all three offset states. The
# default plot shows nominal observations; no offset axis is silently profiled.
exp_plot: vl.ECDFPlot = vl.plot_exp_ecdfs(
    valid_case,
    observable_tag,
    time_idx=2,
    point_idxs=None,
    offset=None,
    view=vl.ValidECDFView(scenario_idxs=None),
)
exp_plot.figure.savefig(output_path / "exp_offset_envelopes.png", bbox_inches="tight")
if not SHOW_PLOTS:
    plt.close(exp_plot.figure)

# Confidence is computed per original experimental sample before enveloping.
sim_bounds: vl.ValidECDFBounds = vl.find_ecdf_epis_bounds(
    valid_case.sim[observable_tag], time_idx=2
)
exp_bounds: vl.ValidECDFBounds = vl.find_ecdf_epis_bounds(
    valid_case.exp[observable_tag], time_idx=2, offset=None
)
confidence_supports: vl.ValidSupportBounds = vl.calc_mavm_over_epis_bounds(
    sim_bounds, exp_bounds
)
(output_path / "confidence_supports.json").write_text(
    json.dumps(
        {
            "probabilities": confidence_supports.probabilities.tolist(),
            "lower": confidence_supports.lower.tolist(),
            "upper": confidence_supports.upper.tolist(),
        },
        indent=2,
    )
)

# %%
# 3. Directional integration figures identify the experimental confidence bounds.
# Select one offset at a time; each time history remains a conditional result.
for offset in vl.EEpisOffset:
    for metric_output, direction in (
        (vl.EMetricOut.D_PLUS, "d_plus"),
        (vl.EMetricOut.D_MINUS, "d_minus"),
    ):
        ecdf_plot: vl.ECDFPlot = vl.plot_sim_exp_cdfs(
            valid_case, observable_tag, point_idxs=None, time_idx=2, exp_offset=offset
        )
        vl.add_metric_overlay(
            ecdf_plot,
            vl.MetricMAVM(),
            output=metric_output,
            display=vl.EMetricDisp.INTEGRATION_AREA,
        )
        ecdf_plot.figure.savefig(
            output_path / f"mavm_{offset.name.lower()}_{direction}_area.png",
            bbox_inches="tight",
        )
        if not SHOW_PLOTS:
            plt.close(ecdf_plot.figure)

for metric_output, direction in (
    (vl.EMetricOut.D_PLUS, "d_plus"),
    (vl.EMetricOut.D_MINUS, "d_minus"),
):
    fig: Figure
    metric_axes: np.ndarray
    fig, metric_axes = vl.plot_metric_time_points(
        valid_result,
        observable_tag,
        vl.EMetricType.MAVM,
        point_idxs=(0, 1),
        selectors={vl.EValidResAxis.EXP_OFFSET: vl.EEpisOffset.NOMINAL},
        output=metric_output,
    )
    fig.savefig(output_path / f"mavm_{direction}_time.png", bbox_inches="tight")
    if not SHOW_PLOTS:
        plt.close(fig)
    ecdf_plot = vl.plot_sim_exp_cdfs(
        valid_case, observable_tag, time_idx=2, exp_offset=None
    )
    vl.add_support_overlay(
        ecdf_plot,
        confidence_supports,
        output=metric_output,
        label="Confidence plus offset envelope",
    )
    ecdf_plot.figure.savefig(
        output_path / f"mavm_{direction}_confidence_support.png", bbox_inches="tight"
    )
    if not SHOW_PLOTS:
        plt.close(ecdf_plot.figure)

if SHOW_PLOTS:
    plt.show()
