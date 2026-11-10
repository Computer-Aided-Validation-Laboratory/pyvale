"""Asymmetric coherent measurement offsets on temperature and strain."""

import matplotlib

matplotlib.use("Agg")

import pyvale.valid as val
from pyvale.examples.valid.tools import load_plate_case, save_diagnostic

case = load_plate_case(offsets=True, epistemic=True, transient=True)
result = val.ValidationEngine([val.MetricAVM(), val.MetricMAVM()]).run(case)

fig, _ = val.plot_val_cdf_envelope(
    case, "temperature", source=val.EValSource.EXP, point=5, time=4
)
save_diagnostic(fig, "ex4_val_bounds_envelope.png")

fig, _ = val.plot_val_metric_map(
    result,
    "temperature",
    val.EMetricType.MAVM,
    selectors={
        val.EResultAxis.TIME: 4,
        val.EResultAxis.SIM_TRANS: 0,
        val.EResultAxis.EXP_OFFSET: val.EOffsetScenario.UPPER,
    },
    direction="d_plus",
)
save_diagnostic(fig, "ex4_val_bounds_directional_map.png")

fig, _ = val.plot_val_metric_time(
    result,
    "temperature",
    val.EMetricType.MAVM,
    selectors={
        val.EResultAxis.POINT: 5,
        val.EResultAxis.SIM_TRANS: 0,
        val.EResultAxis.EXP_OFFSET: val.EOffsetScenario.UPPER,
    },
    direction="d_plus",
)
save_diagnostic(fig, "ex4_val_bounds_time.png")
