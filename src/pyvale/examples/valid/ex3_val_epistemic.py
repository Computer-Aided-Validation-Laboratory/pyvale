"""Two sampled plate parameter scenarios, retained without reduction."""

import matplotlib

matplotlib.use("Agg")

import pyvale.valid as val
from pyvale.examples.valid.tools import load_plate_case, save_diagnostic

case = load_plate_case(epistemic=True)
result = val.ValidationEngine([val.MetricAVM(), val.MetricMAVM()]).run(case)

fig, _ = val.plot_val_cdf_envelope(case, "temperature", point=5)
save_diagnostic(fig, "ex3_val_epistemic_envelope.png")

fig, _ = val.plot_val_metric_distribution(result, "temperature", val.EMetricType.AVM)
save_diagnostic(fig, "ex3_val_epistemic_distribution.png")
