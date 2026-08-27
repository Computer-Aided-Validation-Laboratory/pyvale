# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Sensor Library: Standard Thermo-Mechanical Transducer Catalog
================================================================================

`SensorLibrary` provides ready-to-use virtual models of standard experimental
thermo-mechanical transducers with realistic physical spatial/temporal apertures
and built-in error simulator chains:
- `thermocouple()`: Bead point sensor with optional thermal lag and drift.
- `rtd()`: Platinum resistance detector with volumetric bulb averaging.
- `strain_gauge()` / `strain_rosette()`: Rectangular foil gauges with transverse
  sensitivity and gauge factor uncertainties.
- `fbg_fiber()`: Line-averaged optical Fiber Bragg Grating sensor.
- `extensometer()`: Dual knife-edge clip-on displacement/strain sensor.
- `lvdt()`: Directional axial displacement core sensor.
- `load_cell()`: Multi-axis contact patch traction/force cell.
- `heat_flux_meter()`: Circular foil normal heat flux sensor.
- `flow_meter()`: Volumetric fluid velocity pipe flux meter.
- `lidar()` / `pyrometer()`: Non-contact optical standoff & temperature sensors.
- `pressure_gauge()`: Circular diaphragm pressure sensor.
- `accelerometer()`: Directional dynamic inertial displacement probe.

In this example, we demonstrate:
1. Instantiating transducers in ideal ground truth mode
   (`with_meas_errs=False`).
2. Instantiating transducers in realistic experimental uncertainty mode
   (`with_meas_errs=True`).
3. Comparing measured signals against true simulation values.
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

# Synthesize pressure field
sim_data.node_vars["pressure"] = (
    0.1 * sim_data.node_vars["temperature"]
)

# %%
# 2. Instantiate transducers in ideal and noisy modes
# ---------------------------------------------------
pos = np.array([[20.0, 20.0, 0.0]])

# Thermocouple (ideal vs. with thermal lag and noise)
tc_ideal = sens.SensorLibrary.thermocouple(
    sim_data=sim_data, positions=pos, with_meas_errs=False
)
tc_noisy = sens.SensorLibrary.thermocouple(
    sim_data=sim_data, positions=pos, with_meas_errs=True, time_constant=0.08
)

# 3-branch Strain Rosette (0 / 45 / 90 deg)
rosette = sens.SensorLibrary.strain_rosette(
    sim_data=sim_data,
    position=pos,
    angles_deg=(0.0, 45.0, 90.0),
    with_meas_errs=True,
)

# LVDT axial displacement sensor
lvdt = sens.SensorLibrary.lvdt(
    sim_data=sim_data,
    target_position=pos,
    axis=(1.0, 0.0, 0.0),
    spatial_dims=sens.EDim.TWOD,
    disp_keys=("disp_x", "disp_y"),
    with_meas_errs=True,
)

# Diaphragm Pressure Gauge
pressure_gauge = sens.SensorLibrary.pressure_gauge(
    sim_data=sim_data,
    position=pos,
    diaphragm_radius=2.0,
    with_meas_errs=True,
)

meas_tc_ideal = tc_ideal.sim_measurements()
meas_tc_noisy = tc_noisy.sim_measurements()
meas_rosette = rosette.sim_measurements()
meas_lvdt = lvdt.sim_measurements()
meas_press = pressure_gauge.sim_measurements()

times = sim_data.time

# %%
# 3. Visualise transducer signals
# -------------------------------
show_plots: bool = False
output_path = Path("pyvale-output/extsensorsim")
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    2, 2, figsize=(10, 7.5), dpi=100
)

# Panel 1: Thermocouple
ax1.plot(
    times,
    meas_tc_ideal[0, 0, :],
    "k-",
    linewidth=2.0,
    label="Ideal Truth (°C)",
)
ax1.plot(
    times,
    meas_tc_noisy[0, 0, :],
    "C3--",
    linewidth=1.8,
    label="Sensor Library TC (Lag + Noise)",
)
ax1.set_ylabel("Temperature (°C)", fontsize=10)
ax1.set_title("Thermocouple with Measurement Errors", fontsize=11)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8)

# Panel 2: Strain Rosette
ax2.plot(
    times,
    meas_rosette[0, 0, :] * 1e6,
    "C0-",
    linewidth=1.8,
    label="ε(0°)",
)
ax2.plot(
    times,
    meas_rosette[1, 0, :] * 1e6,
    "C1--",
    linewidth=1.8,
    label="ε(45°)",
)
ax2.plot(
    times,
    meas_rosette[2, 0, :] * 1e6,
    "C2-.",
    linewidth=1.8,
    label="ε(90°)",
)
ax2.set_ylabel("Strain (με)", fontsize=10)
ax2.set_title("3-Branch Strain Gauge Rosette", fontsize=11)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8)

# Panel 3: LVDT
ax3.plot(
    times,
    meas_lvdt[0, 0, :],
    "C4-",
    linewidth=2.0,
    label="LVDT Axial Disp (mm)",
)
ax3.set_xlabel("Time (s)", fontsize=10)
ax3.set_ylabel("Displacement (mm)", fontsize=10)
ax3.set_title("LVDT Transducer", fontsize=11)
ax3.grid(True, linestyle=":", alpha=0.6)
ax3.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8)

# Panel 4: Pressure Gauge
ax4.plot(
    times,
    meas_press[0, 0, :],
    "C5-",
    linewidth=2.0,
    label="Diaphragm Pressure (MPa)",
)
ax4.set_xlabel("Time (s)", fontsize=10)
ax4.set_ylabel("Pressure (MPa)", fontsize=10)
ax4.set_title("Diaphragm Pressure Gauge", fontsize=11)
ax4.grid(True, linestyle=":", alpha=0.6)
ax4.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8)

fig.tight_layout()

save_fig = output_path / "ext_ex8f_sensor_library.png"
fig.savefig(save_fig, dpi=200, bbox_inches="tight")
print(f"Saved plot to: {save_fig}")

# %%
# .. image:: ../../../../_static/ext_ex8f_sensor_library.png
#    :alt: Standard Sensor Library Transducer Suite
#    :width: 700px
#    :align: center

if show_plots:
    plt.show()
else:
    plt.close(fig)
