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

from pyvale import valid
from pyvale.valid import validplot

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
    valid.EValidAxis.POINT,
    valid.EValidAxis.COMPONENT,
    valid.EValidAxis.TIME,
    valid.EValidAxis.ALEATORY,
)
observable_tag = "temperature"
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
            exp_samples,
            axes,
            valid.EpistemicOffsets(
                lower_offset,
                upper_offset,
                (valid.EValidAxis.POINT, valid.EValidAxis.TIME),
            ),
        )
    },
)
valid_result: valid.ValidResult = valid.ValidEngine(
    [valid.MetricAVM(), valid.MetricMAVM()]
).run(valid_case)

# %%
# 2. Only the explicit exp_offset=None expands all three offset states. The
# default plot shows nominal observations; no offset axis is silently profiled.
exp_plot: validplot.ValidECDFPlot = validplot.plot_exp_ecdfs(
    valid_case,
    observable_tag,
    time_idx=2,
    point_idxs=None,
    offset=None,
    view=valid.ValidECDFView(scenario_idxs=None),
)
exp_plot.figure.savefig(output_path / "exp_offset_envelopes.png", bbox_inches="tight")
if not SHOW_PLOTS:
    plt.close(exp_plot.figure)

# Confidence is computed per original experimental sample before enveloping.
sim_bounds: valid.ValidECDFBounds = valid.find_ecdf_epis_bounds(
    valid_case.sim[observable_tag], time_idx=2
)
exp_bounds: valid.ValidECDFBounds = valid.find_ecdf_epis_bounds(
    valid_case.exp[observable_tag], time_idx=2, offset=None
)
confidence_supports: valid.ValidSupportBounds = valid.calc_mavm_over_epis_bounds(
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
for offset in valid.EEpisOffset:
    for metric_output, direction in (
        (valid.EMetricOutput.D_PLUS, "d_plus"),
        (valid.EMetricOutput.D_MINUS, "d_minus"),
    ):
        ecdf_plot: validplot.ValidECDFPlot = validplot.plot_sim_exp_cdfs(
            valid_case, observable_tag, point_idxs=None, time_idx=2, exp_offset=offset
        )
        validplot.add_metric_overlay(
            ecdf_plot,
            valid.MetricMAVM(),
            output=metric_output,
            display=validplot.EValidMetricDisplay.INTEGRATION_AREA,
        )
        ecdf_plot.figure.savefig(
            output_path / f"mavm_{offset.name.lower()}_{direction}_area.png",
            bbox_inches="tight",
        )
        if not SHOW_PLOTS:
            plt.close(ecdf_plot.figure)

for metric_output, direction in (
    (valid.EMetricOutput.D_PLUS, "d_plus"),
    (valid.EMetricOutput.D_MINUS, "d_minus"),
):
    fig: Figure
    metric_axes: np.ndarray
    fig, metric_axes = validplot.plot_valid_metric_time_points(
        valid_result,
        observable_tag,
        valid.EMetricType.MAVM,
        point_idxs=(0, 1),
        selectors={valid.EValidResAxis.EXP_OFFSET: valid.EEpisOffset.NOMINAL},
        output=metric_output,
    )
    fig.savefig(output_path / f"mavm_{direction}_time.png", bbox_inches="tight")
    if not SHOW_PLOTS:
        plt.close(fig)
    ecdf_plot = validplot.plot_sim_exp_cdfs(
        valid_case, observable_tag, time_idx=2, exp_offset=None
    )
    validplot.add_support_overlay(
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
