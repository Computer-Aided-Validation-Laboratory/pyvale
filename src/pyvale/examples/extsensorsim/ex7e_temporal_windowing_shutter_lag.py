# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Temporal Windowing: Exposure Shutter Duration & Sensor Dynamic Lag
================================================================================

In high-speed and transient experimental dynamics, sensors do not sample
instantaneously:
1. High-Speed Cameras: Integrate light over a finite shutter exposure duration
   $[t_0 - \Delta t/2, t_0 + \Delta t/2]$ (`TemporalWindowCentered`).
2. Thermocouples / RTDs: Exhibit first-order thermal inertia (time constant
   $\tau_{\text{RC}}$) with causal historical memory $[t_0 - \Delta t, t_0]$
   (`TemporalWindowCausal` + `TemporalKernelExponentialDecay`).

In this example, we demonstrate:
1. Configuring temporal integration windows with `TemporalWindowCentered` and
   `TemporalWindowCausal`.
2. Modeling sensor dynamic response lag on transient thermal simulation data.
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

# %%
# 2. Configure temporal sensors
# -----------------------------
pos = np.array([[10.0, 15.0, 0.0]])
sample_times = np.linspace(0.1, 1.0, 20)

sens_data = sens.SensorData(positions=pos, sample_times=sample_times)
field = sens.FieldScalar(
    sim_data=sim_data,
    comp_key="temperature",
    spatial_dims=sens.EDim.TWOD,
)

# A: Ideal instantaneous sampling (shutter time = 0)
sensor_instant = sens.SensorsSpatial(
    sensor_data=sens_data,
    field=field,
    temporal_window=sens.TemporalWindowInstant(),
)

# B: Camera shutter exposure (centered window duration = 0.08 s)
sensor_shutter = sens.SensorsSpatial(
    sensor_data=sens_data,
    field=field,
    temporal_window=sens.TemporalWindowCentered(duration=0.08),
)

# C: Thermocouple with thermal lag (tau = 0.05 s, duration = 0.15 s)
sensor_lag = sens.SensorsSpatial(
    sensor_data=sens_data,
    field=field,
    temporal_window=sens.TemporalWindowCausal(
        duration=0.15,
        kernel=sens.TemporalKernelExponentialDecay(time_constant=0.05),
    ),
)

# %%
# 3. Simulate measurements
# ------------------------
meas_inst = sensor_instant.sim_measurements()
meas_shut = sensor_shutter.sim_measurements()
meas_lag = sensor_lag.sim_measurements()

# %%
# 4. Plot temporal filtering effects
# ----------------------------------
show_plots: bool = False

output_path = Path.cwd() / "pyvale-output" / "extsensorsim"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

ax.plot(
    sample_times,
    meas_inst[0, 0, :],
    "k--o",
    label="Instantaneous (Δt = 0)",
)
ax.plot(
    sample_times,
    meas_shut[0, 0, :],
    "C0-s",
    label="Exposure Shutter (Δt = 0.08 s)",
)
ax.plot(
    sample_times,
    meas_lag[0, 0, :],
    "C3-^",
    label="Thermal Lag (τ = 0.05 s)",
)

ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel("Temperature (°C)", fontsize=11)
ax.set_title(
    "Temporal Windowing: Exposure Shutter & Dynamic Lag", fontsize=12
)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(frameon=True, facecolor="white")
fig.tight_layout()

save_fig = output_path / "ext_ex7e_temporal_shutter_lag.png"
fig.savefig(save_fig, dpi=200, bbox_inches="tight")
print(f"Saved plot to: {save_fig}")

# %%
# .. image:: ../../../../_static/ext_ex7e_temporal_shutter_lag.png
#    :alt: Temporal shutter duration and dynamic lag filter
#    :width: 700px
#    :align: center

if show_plots:
    plt.show()
else:
    plt.close(fig)
