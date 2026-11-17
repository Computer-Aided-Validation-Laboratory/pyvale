# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Steady transferable bounds versus comparisons of every original scenario.

Crossing conditional ECDFs make the two routes visibly different. This example
has no nuisance uncertainty and needs no downloaded data. Bounds remain curves
with empirical probability mass, rather than synthetic equally weighted samples.
"""

import json
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
output_path = Path.cwd() / "pyvale-output" / "valid_ex4_transferable_bounds"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Construct crossing sim ECDFs and two transferable exp alternatives.
# At p=.5 the second sim case supplies the upper bound; at p=1 the first does.
# Each row is a complete conditional aleatory sample, never pooled with another.
observable_tag = "response"
axes = (valid.EValidAxis.EPISTEMIC_TRANS, valid.EValidAxis.ALEATORY)
valid_case = valid.ValidCase(
    {
        observable_tag: valid.ValidObservable(
            components=("response",),
            geometry=valid.EValidGeometry.GLOBAL,
            phys_units="unit",
        )
    },
    {observable_tag: valid.ValidArray(np.array([[0.0, 100.0], [40.0, 60.0]]), axes)},
    {observable_tag: valid.ValidArray(np.array([[-10.0, 0.0], [100.0, 110.0]]), axes)},
)

# %%
# 2. The boundary route needs no engine. Its four comparisons use pointwise
# envelope boundaries that can splice together different original scenarios.
sim_bounds: valid.ValidECDFBounds = valid.find_ecdf_epis_bounds(
    valid_case.sim[observable_tag]
)
exp_bounds: valid.ValidECDFBounds = valid.find_ecdf_epis_bounds(
    valid_case.exp[observable_tag]
)
avm_bounds: valid.ValidSupportBounds = valid.calc_avm_over_epis_bounds(
    sim_bounds, exp_bounds
)
# alpha=.5 deliberately removes confidence shifts to isolate epistemic geometry.
# ex07 demonstrates default MAVM confidence construction and coherent offsets.
mavm_bounds: valid.ValidSupportBounds = valid.calc_mavm_over_epis_bounds(
    sim_bounds, exp_bounds, alpha=0.5
)

# 3. The original-scenario route calculates all comparisons, then reuses them.
valid_result: valid.ValidResult = valid.ValidEngine(
    [valid.MetricAVM(), valid.MetricMAVM(alpha=0.5)]
).run(valid_case)
avm_original: valid.ValidSupportBounds = valid.find_metric_epis_supports(
    valid_case, valid_result, observable_tag, valid.EMetricType.AVM
)
mavm_original: valid.ValidSupportBounds = valid.find_metric_epis_supports(
    valid_case, valid_result, observable_tag, valid.EMetricType.MAVM
)
extrema: valid.ValidMetricExtrema = valid.find_metric_epis_extrema(
    valid_result, observable_tag, valid.EMetricType.MAVM
)
# Explicit analytic answers: original scenario shifts reach [-55,5] and [95,155].
# The synthetic boundary comparisons instead reach [-35,25] and [75,135].
np.testing.assert_allclose(mavm_original.lower[0, 0, 0], [-55.0, 5.0], atol=1.0e-12)
np.testing.assert_allclose(mavm_original.upper[0, 0, 0], [95.0, 155.0], atol=1.0e-12)
np.testing.assert_allclose(mavm_bounds.lower, [-35.0, 25.0], atol=1.0e-12)
np.testing.assert_allclose(mavm_bounds.upper, [75.0, 135.0], atol=1.0e-12)
(output_path / "comparison.json").write_text(
    json.dumps(
        {
            "sim_lower_origins": sim_bounds.lower_indices.tolist(),
            "sim_upper_origins": sim_bounds.upper_indices.tolist(),
            "original_index_axes": [axis.name for axis in mavm_original.index_axes],
            "original_upper_origins": mavm_original.upper_indices[0, 0, 0].tolist(),
            "bounds_lower": mavm_bounds.lower.tolist(),
            "bounds_upper": mavm_bounds.upper.tolist(),
            "original_lower": mavm_original.lower[0, 0, 0].tolist(),
            "original_upper": mavm_original.upper[0, 0, 0].tolist(),
            "maximum_d_plus": float(
                extrema.maximum[valid.EMetricOutput.D_PLUS][0, 0, 0]
            ),
            "maximum_d_minus": float(
                extrema.maximum[valid.EMetricOutput.D_MINUS][0, 0, 0]
            ),
        },
        indent=2,
    )
)

# %%
# 4. Compare the two definitions on separate figures, with the same original
# scenario curves available for context. Neither route is relabelled as the other.
for route, avm, mavm in (
    ("bounds", avm_bounds, mavm_bounds),
    ("original", avm_original, mavm_original),
):
    ecdf_plot: validplot.ValidECDFPlot = validplot.plot_sim_exp_cdfs(
        valid_case, observable_tag
    )
    validplot.add_support_overlay(ecdf_plot, avm, label=f"AVM symmetric shift: {route}")
    ecdf_plot.figure.savefig(output_path / f"avm_{route}.png", bbox_inches="tight")
    if not SHOW_PLOTS:
        plt.close(ecdf_plot.figure)
    for metric_output, direction in (
        (valid.EMetricOutput.D_PLUS, "d_plus"),
        (valid.EMetricOutput.D_MINUS, "d_minus"),
    ):
        ecdf_plot = validplot.plot_sim_exp_cdfs(valid_case, observable_tag)
        validplot.add_support_overlay(
            ecdf_plot, mavm, output=metric_output, label=f"MAVM {direction}: {route}"
        )
        ecdf_plot.figure.savefig(
            output_path / f"mavm_{direction}_{route}.png", bbox_inches="tight"
        )
        if not SHOW_PLOTS:
            plt.close(ecdf_plot.figure)

if SHOW_PLOTS:
    plt.show()
