# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Experiment Simulator: Multi-Transducer Monte Carlo Uncertainty Analysis
================================================================================

In rigorous digital twin validation campaigns, virtual experiments combine
heterogeneous transducers (thermocouples, strain gauges, and derived field
probes) across multi-run Monte Carlo trials to characterize sensor uncertainty
propagation and measurement confidence bounds.

In this example, we demonstrate:
1. Combining transducers from `SensorLibrary` with `FieldTransformed` derived
   sensors into an `ExperimentSimulator`.
2. Executing a multi-trial Monte Carlo virtual experiment.
3. Calculating statistical means and 95% confidence intervals with
   `calc_exp_sim_stats()`.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# pyvale imports
import pyvale.sensorsim as sens
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.data as dataset


# %%
# 1. Load thermo-mechanical simulation data
# -----------------------------------------
data_path: Path = dataset.thermomechanical_2d_path()
sim_data: io.SimData = mh.ExodusLoader(data_path).load_all_sim_data()
sim_data = sens.scale_length_units(
    scale=1000.0, sim_data=sim_data, disp_keys=None
)

# %%
# 2. Deploy heterogeneous transducer suite with error models
# ----------------------------------------------------------
probe_pos = np.array([[20.0, 20.0, 0.0]])

# 1. Thermocouple with thermal lag and calibration uncertainty
tc_sensor = sens.SensorLibrary.thermocouple(
    sim_data=sim_data,
    positions=probe_pos,
    with_meas_errs=True,
    time_constant=0.05,
)

# 2. Foil strain gauge with gauge factor and transverse errors
sg_sensor = sens.SensorLibrary.strain_gauge(
    sim_data=sim_data,
    positions=probe_pos,
    spatial_dims=sens.EDim.TWOD,
    with_meas_errs=True,
)

# %%
# 3. Assemble and execute Experiment Simulator
# --------------------------------------------
sim_dict = {"thermomech": sim_data}
sensor_arrays = {"thermocouple": tc_sensor, "strain_gauge": sg_sensor}
exp_sim_opts = sens.ExpSimOpts(workers=1, para=sens.EExpSimPara.ALL)

exp_sim = sens.ExperimentSimulator(
    sim_dict=sim_dict,
    sensor_arrays=sensor_arrays,
    exp_sim_opts=exp_sim_opts,
)

num_runs = 25
exp_data = exp_sim.run_experiments(num_exp_per_sim=num_runs)
stats = sens.calc_exp_sim_stats(exp_data)

# %%
# 4. Visualise Monte Carlo confidence intervals
# ---------------------------------------------
show_plots: bool = False
output_path = Path("pyvale-output/extsensorsim")
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

times = sim_data.time
stats_tc = stats[("thermomech", "thermocouple", "meas")]
stats_sg = stats[("thermomech", "strain_gauge", "meas")]

mean_tc = stats_tc.mean[0, 0, :]
std_tc = stats_tc.std[0, 0, :]
truth_tc = tc_sensor.get_truth()[0, 0, :]

mean_sg = stats_sg.mean[0, 0, :] * 1e6
std_sg = stats_sg.std[0, 0, :] * 1e6
truth_sg = sg_sensor.get_truth()[0, 0, :] * 1e6

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=100)

# Panel 1: Thermocouple MC statistics
ax1.plot(times, truth_tc, "k--", linewidth=2.0, label="Simulation Truth")
ax1.plot(times, mean_tc, "C3-", linewidth=1.8, label="MC Mean")
ax1.fill_between(
    times,
    mean_tc - 1.96 * std_tc,
    mean_tc + 1.96 * std_tc,
    color="C3",
    alpha=0.25,
    label="95% Confidence Interval",
)
ax1.set_xlabel("Time (s)", fontsize=11)
ax1.set_ylabel("Temperature (°C)", fontsize=11)
ax1.set_title("Thermocouple Monte Carlo Analysis", fontsize=12)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9)

# Panel 2: Strain Gauge MC statistics
ax2.plot(times, truth_sg, "k--", linewidth=2.0, label="Simulation Truth")
ax2.plot(times, mean_sg, "C0-", linewidth=1.8, label="MC Mean")
ax2.fill_between(
    times,
    mean_sg - 1.96 * std_sg,
    mean_sg + 1.96 * std_sg,
    color="C0",
    alpha=0.25,
    label="95% Confidence Interval",
)
ax2.set_xlabel("Time (s)", fontsize=11)
ax2.set_ylabel("Strain (με)", fontsize=11)
ax2.set_title("Strain Gauge Monte Carlo Analysis", fontsize=12)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9)

fig.tight_layout()

save_fig = output_path / "ext_ex8g_expsim_montecarlo.png"
fig.savefig(save_fig, dpi=200, bbox_inches="tight")
print(f"Saved plot to: {save_fig}")

# %%
# .. image:: ../../../../_static/ext_ex8g_expsim_montecarlo.png
#    :alt: Multi-Transducer Monte Carlo Uncertainty Analysis
#    :width: 700px
#    :align: center

if show_plots:
    plt.show()
else:
    plt.close(fig)
