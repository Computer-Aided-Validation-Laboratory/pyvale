# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Volume Sensors: 3D RTD Probe & Accumulate Mode
================================================================================

Many physical transducers have volumetric spatial extent (such as Resistance
Temperature Detector RTD platinum bulbs or calorimeter cells).

Furthermore, physical quantities are categorized as:
1. Intensive (Average mode): Temperature (°C), strain, stress, pressure.
   Weights normalize to 1.0.
2. Extensive (Accumulate mode): Total integrated thermal energy, heat capacity
   integral, or total volume flux. Weights equal the physical volume ($mm^3$).

In this example, we demonstrate:
1. Creating 3D volume sensors with `SpatialWindowBox` and
   `SpatialWindowCylinder`.
2. Setting `EIntegrationMode.AVERAGE` vs. `EIntegrationMode.ACCUMULATE`.
3. Verifying that Accumulate $\equiv$ Average $\times$ Volume.
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
# 1. Load 3D thermal simulation data
# ----------------------------------
data_path: Path = dataset.thermal_3d_path()
sim_data: io.SimData = mh.ExodusLoader(data_path).load_all_sim_data()
sim_data = sens.scale_length_units(
    scale=1000.0, sim_data=sim_data, disp_keys=None
)

# %%
# 2. Configure 3D volume sensors
# ------------------------------
# Probe center in the interior of the 3D block
probe_pos = np.array([[25.0, 25.0, 10.0]])
sens_data = sens.SensorData(
    positions=probe_pos,
    sample_times=sim_data.time,
)

field = sens.FieldScalar(
    sim_data=sim_data,
    comp_key="temperature",
    spatial_dims=sens.EDim.THREED,
)

# 3D cuboid RTD bulb: 6 mm x 6 mm x 4 mm (Volume = 144 mm^3)
rtd_box = sens.SpatialWindowBox(
    length_x=6.0,
    length_y=6.0,
    length_z=4.0,
    integ_rule=sens.IntegrationGaussLegendre(order=2),
)

# Sensor in AVERAGE mode (Intensive: Mean temperature in °C)
sensor_rtd_avg = sens.SensorsSpatial(
    sensor_data=sens_data,
    field=field,
    spatial_window=rtd_box,
    integration_mode=sens.EIntegrationMode.AVERAGE,
)

# Sensor in ACCUMULATE mode (Extensive: Volume integral in °C * mm^3)
sensor_rtd_acc = sens.SensorsSpatial(
    sensor_data=sens_data,
    field=field,
    spatial_window=rtd_box,
    integration_mode=sens.EIntegrationMode.ACCUMULATE,
)

# Point sensor at probe center for comparison
sensor_point = sens.SensorFactory.scalar_point(
    sim_data=sim_data,
    sensor_data=sens_data,
    comp_key="temperature",
    spatial_dims=sens.EDim.THREED,
)

# %%
# 3. Simulate measurements
# ------------------------
meas_avg = sensor_rtd_avg.sim_measurements()  # (°C)
meas_acc = sensor_rtd_acc.sim_measurements()  # (°C * mm^3)
meas_pt = sensor_point.sim_measurements()

volume = rtd_box.get_measure()  # 144.0 mm^3
times = sensor_rtd_avg.get_sample_times()

# Verify mathematical consistency
assert np.allclose(meas_acc, meas_avg * volume)

# %%
# 4. Visualise comparison
# -----------------------
show_plots: bool = False

output_path = Path.cwd() / "pyvale-output" / "extsensorsim"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=100)

ax1.plot(
    times,
    meas_pt[0, 0, :],
    "k--",
    label="Point Sensor (0D)",
)
ax1.plot(
    times,
    meas_avg[0, 0, :],
    "C0-",
    linewidth=2.0,
    label=r"RTD Bulb Average ($V=144\,\text{mm}^3$)",
)
ax1.set_xlabel("Time (s)", fontsize=11)
ax1.set_ylabel("Temperature (°C)", fontsize=11)
ax1.set_title("Average Mode (Intensive Quantity)", fontsize=12)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(frameon=True)

ax2.plot(
    times,
    meas_acc[0, 0, :],
    "C3-",
    linewidth=2.0,
    label=r"Accumulated $\int T\,dV$",
)
ax2.set_xlabel("Time (s)", fontsize=11)
ax2.set_ylabel(
    r"Integrated Value ($^\circ\text{C}\cdot\text{mm}^3$)", fontsize=11
)
ax2.set_title("Accumulate Mode (Extensive Quantity)", fontsize=12)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(frameon=True)

fig.tight_layout()

save_fig = output_path / "ext_ex7c_volume_rtd.png"
fig.savefig(save_fig, dpi=200, bbox_inches="tight")
print(f"Saved plot to: {save_fig}")

# %%
# .. image:: ../../../../_static/ext_ex7c_volume_rtd.png
#    :alt: Volumetric RTD sensor average vs accumulate modes
#    :width: 700px
#    :align: center

if show_plots:
    plt.show()
else:
    plt.close(fig)
