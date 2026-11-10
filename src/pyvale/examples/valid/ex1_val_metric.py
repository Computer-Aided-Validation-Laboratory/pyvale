"""Standalone validation metrics on hand-checkable NumPy samples.

Each curve is an empirical CDF. The shifted samples have AVM distance 2,
KS distance 2/3 because they share one support value, and paired RMSE 2.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import pyvale.valid as val
from pyvale.examples.valid.tools import save_diagnostic

simulation = np.array([0.0, 1.0, 2.0])
experiment = simulation + 2.0
metrics = (
    val.MetricAVM(),
    val.MetricKS(),
    val.MetricCVM(),
    val.MetricRMSE(),
    val.MetricMAVM(),
)
for metric in metrics:
    result = metric.calc(simulation, experiment)
    print(f"{result.kind.name}: {result.value:.6g}")

fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
for values, label in ((simulation, "Simulation"), (experiment, "Experiment")):
    cdf = stats.ecdf(values).cdf
    ax.step(cdf.quantiles, cdf.probabilities, where="post", label=label)
ax.set(
    xlabel="Observable [unit]",
    ylabel="Cumulative probability",
    title="Shifted empirical distributions: AVM = 2, KS = 2/3",
)
ax.legend()
ax.grid(alpha=0.3)
save_diagnostic(fig, "ex1_val_metric_cdfs.png")
