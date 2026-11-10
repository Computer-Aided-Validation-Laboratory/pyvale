"""Steady 2D plate: prepared arrays through ValCase and ValidationEngine."""

import matplotlib

matplotlib.use("Agg")

import pyvale.valid as val
from pyvale.examples.valid.tools import load_plate_case, save_diagnostic

case = load_plate_case()
engine = val.ValidationEngine([val.MetricAVM(), val.MetricMAVM()])
result = engine.run(case)

fig, _ = val.plot_val_cdfs(
    case, result, "temperature", point=5, metric=val.MetricMAVM()
)
save_diagnostic(fig, "ex2_val_steady_cdfs.png")

fig, _ = val.plot_val_metric_map(result, "temperature", val.EMetricType.AVM)
save_diagnostic(fig, "ex2_val_steady_map.png")
