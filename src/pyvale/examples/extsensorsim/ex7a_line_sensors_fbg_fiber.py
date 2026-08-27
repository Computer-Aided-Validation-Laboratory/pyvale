# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Line Sensors: Optical Fiber Bragg Grating (FBG)
================================================================================

In physical experimental mechanics, optical fiber sensors (e.g. Fiber Bragg
Gratings, FBG, or Distributed Temperature/Strain Sensing, DTS) measure physical
fields integrated continuously along a 1D line of finite gauge length.

In this example, we demonstrate:
1. Creating 1D line sensors with `SensorsSpatial` and `SpatialWindowLine`.
2. Constructing a line sensor directly between two 3D endpoints using
   `SensorFactory.line_from_endpoints()`.
3. Spatial averaging along fiber gauge length using Gauss-Legendre
   quadrature.
4. Comparing a finite gauge-length line sensor with an idealized point sensor
   across a high thermal gradient.
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
# We load a 2D transient thermal finite element simulation.
data_path: Path = dataset.thermal_2d_path()
sim_data: io.SimData = mh.ExodusLoader(data_path).load_all_sim_data()
sim_data = sens.scale_length_units(
    scale=1000.0, sim_data=sim_data, disp_keys=None
)

# %%
# 2. Define optical fiber line sensors across a thermal gradient
# --------------------------------------------------------------
# We place two fiber-optic sensor segments:
# Fiber A: 10 mm gauge length along X
# Fiber B: 20 mm gauge length along X
p_start_a = np.array([5.0, 15.0, 0.0])
p_end_a = np.array([15.0, 15.0, 0.0])

fiber_a = sens.SensorFactory.line_from_endpoints(
    sim_data=sim_data,
    point_start=p_start_a,
    point_end=p_end_a,
    comp_key="temperature",
    spatial_dims=sens.EDim.TWOD,
)

p_start_b = np.array([0.0, 15.0, 0.0])
p_end_b = np.array([20.0, 15.0, 0.0])

fiber_b = sens.SensorFactory.line_from_endpoints(
    sim_data=sim_data,
    point_start=p_start_b,
    point_end=p_end_b,
    comp_key="temperature",
    spatial_dims=sens.EDim.TWOD,
)

# Idealized point sensor at the center (10, 15, 0)
center_pos = np.array([[10.0, 15.0, 0.0]])
sens_data_pt = sens.SensorData(positions=center_pos)
point_sensor = sens.SensorFactory.scalar_point(
    sim_data=sim_data,
    sensor_data=sens_data_pt,
    comp_key="temperature",
    spatial_dims=sens.EDim.TWOD,
)

# %%
# 3. Simulate measurements
# ------------------------
meas_fiber_a = fiber_a.sim_measurements()  # (1, 1, n_times)
meas_fiber_b = fiber_b.sim_measurements()
meas_point = point_sensor.sim_measurements()

times = fiber_a.get_sample_times()

# %%
# 4. Plot comparison of gauge lengths
# -----------------------------------
show_plots: bool = False

output_path = Path.cwd() / "pyvale-output" / "extsensorsim"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
ax.plot(
    times,
    meas_point[0, 0, :],
    "k--",
    linewidth=2.0,
    label="Ideal Point Sensor (0 mm)",
)
ax.plot(
    times,
    meas_fiber_a[0, 0, :],
    "C0-",
    linewidth=2.0,
    label="FBG Fiber A (10 mm gauge)",
)
ax.plot(
    times,
    meas_fiber_b[0, 0, :],
    "C3-.",
    linewidth=2.0,
    label="FBG Fiber B (20 mm gauge)",
)

ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel("Temperature (°C)", fontsize=11)
ax.set_title("Optical Fiber Line Sensor vs. Point Sensor", fontsize=12)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(frameon=True, facecolor="white", edgecolor="none")
fig.tight_layout()

save_fig = output_path / "ext_ex7a_fiber_lines.png"
fig.savefig(save_fig, dpi=200, bbox_inches="tight")
print(f"Saved plot to: {save_fig}")

# %%
# 5. 3D Sensor Geometry Visualisation with PyVista
# ------------------------------------------------
vis_opts = sens.VisOptsSimSensors(interactive=show_plots)
geom_opts = sens.VisOptsSensorGeom(line_radius=0.75, color_nominal="red")
img_save = sens.VisOptsImageSave(
    path=output_path / "ext_ex7a_fiber_3d_geom.png"
)
pv_plot = sens.plot_sensors_on_sim(
    sensor_array=fiber_a,
    component="temperature",
    vis_opts=vis_opts,
    geom_opts=geom_opts,
    image_save_opts=img_save,
)
if show_plots:
    pv_plot.show()
else:
    pv_plot.close()

# %%
# .. image:: ../../../../_static/ext_ex7a_fiber_lines.png
#    :alt: Optical fiber line sensor measurement comparison
#    :width: 700px
#    :align: center

if show_plots:
    plt.show()
else:
    plt.close(fig)
