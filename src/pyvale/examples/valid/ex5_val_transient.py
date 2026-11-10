"""Time-resolved validation on shared plate evaluation times."""

import matplotlib

matplotlib.use("Agg")

import pyvale.valid as val
from pyvale.examples.valid.tools import load_plate_case, save_diagnostic

case = load_plate_case(transient=True)
result = val.ValidationEngine([val.MetricAVM(), val.MetricMAVM()]).run(case)

fig, _ = val.plot_val_metric_time(
    result,
    "temperature",
    val.EMetricType.AVM,
    selectors={val.EResultAxis.POINT: 5},
)
save_diagnostic(fig, "ex5_val_transient_time.png")

fig, _ = val.plot_val_cdfs(
    case,
    result,
    "temperature",
    point=5,
    time=3,
    metric=val.MetricMAVM(),
)
save_diagnostic(fig, "ex5_val_transient_cdfs.png")
