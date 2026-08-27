# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Experiment Simulator: Multi-Array Spatio-Temporal Monte Carlo Analysis
================================================================================

In full-scale digital shadow and virtual validation applications, an
experimental campaign employs multiple heterogeneous sensor arrays
(e.g. line fiber sensors, surface area gauges, and probe volume sensors).

In this example, we demonstrate:
1. Combining multiple spatio-temporal sensor arrays into `ExperimentSimulator`.
2. Running a Monte Carlo virtual campaign to evaluate multi-sensor statistics.
3. Calculating mean and 95% confidence intervals with `calc_exp_sim_stats()`.
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
# 1. Load physics simulation data
# -------------------------------
data_path: Path = dataset.thermal_2d_path()
sim_data: io.SimData = mh.ExodusLoader(data_path).load_all_sim_data()
sim_data = sens.scale_length_units(
    scale=1000.0, sim_data=sim_data, disp_keys=None
)

field = sens.FieldScalar(
    sim_data=sim_data,
    comp_key="temperature",
    spatial_dims=sens.EDim.TWOD,
)

# %%
# 2. Build heterogeneous sensor arrays with spatio-temporal windows
# -----------------------------------------------------------------
# Array 1: 1D Line Sensor (FBG optical fiber)
pos_line = np.array([[10.0, 15.0, 0.0]])
sens_data_line = sens.SensorData(
    positions=pos_line, sample_times=sim_data.time
)
window_line = sens.SpatialWindowLine(length=8.0)
sensor_line = sens.SensorsSpatial(
    sensor_data=sens_data_line,
    field=field,
    spatial_window=window_line,
    descriptor=sens.SensorDescriptor(name="FBG Fiber"),
)
sensor_line.set_error_chain([
    sens.ErrRandGen(generator=sens.GenNormal(std=0.2, mean=0.0))
])

# Array 2: 2D Area Sensor (Thermopile patch)
pos_area = np.array([[18.0, 12.0, 0.0]])
sens_data_area = sens.SensorData(
    positions=pos_area, sample_times=sim_data.time
)
window_area = sens.SpatialWindowRectangle(length_x=4.0, length_y=4.0)
sensor_area = sens.SensorsSpatial(
    sensor_data=sens_data_area,
    field=field,
    spatial_window=window_area,
    descriptor=sens.SensorDescriptor(name="Thermopile"),
)
sensor_area.set_error_chain([
    sens.ErrRandGen(generator=sens.GenUniform(low=-0.3, high=0.3))
])

# %%
# 3. Assemble Experiment Simulator
# --------------------------------
sim_dict = {"thermal": sim_data}
sensor_arrays = {"fiber": sensor_line, "thermopile": sensor_area}
exp_sim_opts = sens.ExpSimOpts(workers=1, para=sens.EExpSimPara.ALL)

exp_sim = sens.ExperimentSimulator(
    sim_dict=sim_dict,
    sensor_arrays=sensor_arrays,
    exp_sim_opts=exp_sim_opts,
)

# %%
# 4. Run Monte Carlo simulation campaign
# --------------------------------------
num_runs = 30
exp_data = exp_sim.run_experiments(num_exp_per_sim=num_runs)
stats = sens.calc_exp_sim_stats(exp_data)

# %%
# 5. Visualise Monte Carlo statistics
# -----------------------------------
show_plots: bool = False

output_path = Path.cwd() / "pyvale-output" / "extsensorsim"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

times = sensor_line.get_sample_times()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=100)

# Fiber statistics
stats_fiber = stats[("thermal", "fiber", "meas")]
mean_f = stats_fiber.mean[0, 0, :]
std_f = stats_fiber.std[0, 0, :]
truth_f = sensor_line.get_truth()[0, 0, :]

ax1.plot(times, truth_f, "k--", label="Truth")
ax1.plot(times, mean_f, "C0-", label="MC Mean")
ax1.fill_between(
    times,
    mean_f - 2.0 * std_f,
    mean_f + 2.0 * std_f,
    color="C0",
    alpha=0.25,
    label=r"$\pm 2\sigma$ Bounds",
)
ax1.set_xlabel("Time (s)", fontsize=11)
ax1.set_ylabel("Temperature (°C)", fontsize=11)
ax1.set_title("Line Sensor (FBG Fiber)", fontsize=12)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(frameon=True)

# Thermopile statistics
stats_thermo = stats[("thermal", "thermopile", "meas")]
mean_t = stats_thermo.mean[0, 0, :]
std_t = stats_thermo.std[0, 0, :]
truth_t = sensor_area.get_truth()[0, 0, :]

ax2.plot(times, truth_t, "k--", label="Truth")
ax2.plot(times, mean_t, "C1-", label="MC Mean")
ax2.fill_between(
    times,
    mean_t - 2.0 * std_t,
    mean_t + 2.0 * std_t,
    color="C1",
    alpha=0.25,
    label=r"$\pm 2\sigma$ Bounds",
)
ax2.set_xlabel("Time (s)", fontsize=11)
ax2.set_ylabel("Temperature (°C)", fontsize=11)
ax2.set_title("Area Sensor (Thermopile)", fontsize=12)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(frameon=True)

fig.tight_layout()

save_fig = output_path / "ext_ex7g_spatiotemp_expsim.png"
fig.savefig(save_fig, dpi=200, bbox_inches="tight")
print(f"Saved plot to: {save_fig}")

# %%
# .. image:: ../../../../_static/ext_ex7g_spatiotemp_expsim.png
#    :alt: Monte Carlo spatio-temporal virtual experiment statistics
#    :width: 700px
#    :align: center

if show_plots:
    plt.show()
else:
    plt.close(fig)
