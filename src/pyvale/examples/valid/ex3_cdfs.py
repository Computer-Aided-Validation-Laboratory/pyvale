# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Example 3: Computing and Plotting Empirical CDFs and Epistemic p-boxes.

Demonstrates extracting empirical CDFs from experimental measurements and
computing epistemic p-box boundaries from probabilistic FE simulations.
"""

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.valid as val


# Set to True to display interactive figures as well as saving the plots.
SHOW_FIGURES = False

#%%
# Set up paths and loading options.
if not SHOW_FIGURES:
    matplotlib.use("Agg")

output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

data_path: Path = dataset.valid_data_dir()
load_opts: io.ExpLoadOpts = io.ExpLoadOpts(delimiter=",", header_rows=0)

#%%
# 1. Load experimental data
exp_loaders: dict[str, io.IExpLoader] = {
    "DIC-DAQ": io.PointSensLoader(
        load_files=[
            data_path / "Pulse253_SteadyDICData.csv",
            data_path / "Pulse254_SteadyDICData.csv",
            data_path / "Pulse255_SteadyDICData.csv",
        ],
        sens_cols=np.array([3, 5]),
        sens_labels=["TC3", "TC5"],
        load_opts=load_opts,
    ),
    "HIVE-DAQ": io.PointSensLoader(
        load_files=[
            data_path / "Pulse253_SteadyHIVEData.csv",
            data_path / "Pulse254_SteadyHIVEData.csv",
            data_path / "Pulse255_SteadyHIVEData.csv",
        ],
        sens_cols=np.array([2]),
        sens_labels=["TC2"],
        load_opts=load_opts,
    ),
}
# Each DAQ key maps to an ExpData container with fields, times and sensor labels.
exp_data: dict[str, io.ExpData] = io.load_exp_data(exp_loaders)
# PointValData groups experimental arrays by DAQ: (n_sensors, n_repeats).
exp_val_data: val.PointValData = val.extract_val_data_by_key(exp_data)

#%%
# 2. Load simulation data
sim_csv: Path = data_path / "SamplingResultsOnlyPointSensors.csv"
sens_keys: dict[str, int] = {"TC2": 1, "TC3": 2, "TC5": 4}
# Simulation arrays in PointValData have axes (sensor, epistemic, aleatory).
sim_val_data: val.PointValData = val.load_prob_sim_csv(
    csv_path=sim_csv,
    sens_keys=sens_keys,
    n_epistemic=50,
    n_aleatory=100,
)

#%%
# 3. Compute p-boxes and plot comparison for TC3
pboxes: dict[str, tuple[np.ndarray, np.ndarray]] = val.calc_limit_cdfs_point(
    sim_val_data
)
sim_lower, sim_upper = pboxes["TC3"]
exp_tc3: np.ndarray = exp_val_data.val_points["DIC-DAQ"][0, :].ravel()

#%%
# Calculate empirical CDFs
cdf_sim_low = stats.ecdf(sim_lower).cdf
cdf_sim_up = stats.ecdf(sim_upper).cdf
cdf_exp = stats.ecdf(exp_tc3).cdf

fig: plt.Figure
ax: plt.Axes
fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
ax.step(
    cdf_sim_low.quantiles,
    cdf_sim_low.probabilities,
    where="post",
    color="firebrick",
    linewidth=2,
    label="Sim p-box Min CDF",
)
ax.step(
    cdf_sim_up.quantiles,
    cdf_sim_up.probabilities,
    where="post",
    color="crimson",
    linewidth=2,
    label="Sim p-box Max CDF",
)
ax.step(
    cdf_exp.quantiles,
    cdf_exp.probabilities,
    where="post",
    color="navy",
    linewidth=2,
    label="Exp Empirical CDF",
)

ax.set_title("Thermocouple TC3: Empirical CDFs and Epistemic p-box")
ax.set_xlabel(r"Temperature [$^\circ\text{C}$]")
ax.set_ylabel("Cumulative Probability")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="best")

save_path: Path = output_path / "ex3_cdfs_output.png"
fig.savefig(save_path, dpi=200)
print(f"Saved CDF comparison plot to: {save_path}")

if SHOW_FIGURES:
    plt.show()
plt.close(fig)
