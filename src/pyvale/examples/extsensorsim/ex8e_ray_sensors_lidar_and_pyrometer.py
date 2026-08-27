# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Ray-Casting Sensors: Optical LIDAR and Infrared Pyrometers
================================================================================

Non-contact optical transducers project sightlines or laser beams into the
experimental test domain:
1. LIDAR Distance Sensors: Measure standoff distance
   :math:`d(t) = \\|\\mathbf{x}_{\\text{hit}}(t) - \\mathbf{x}_0\\|`
   by intersecting a beam ray with the dynamic, deforming specimen surface.
2. Infrared Surface Pyrometers: Sample the surface temperature field
   :math:`T(\\mathbf{x}_{\\text{hit}}(t), t)` at the exact optical strike point.

In this example, we demonstrate:
1. Ray tracing against finite element boundary surfaces with `SensorsRay`.
2. Standoff distance tracking with `SensorLibrary.lidar()`.
3. Non-contact optical temperature monitoring with `SensorLibrary.pyrometer()`.
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
# 2. Deploy LIDAR standoff distance sensor and optical pyrometer
# --------------------------------------------------------------
scanner_pos = np.array([25.0, 25.0, 100.0])
beam_dir = np.array([0.0, 0.0, -1.0])

disp_field = sens.FieldVector(
    sim_data, ("disp_x", "disp_y"), sens.EDim.TWOD
)

lidar_sensor = sens.SensorLibrary.lidar(
    sim_data=sim_data,
    scanner_position=scanner_pos,
    beam_direction=beam_dir,
    disp_field=disp_field,
    max_range=200.0,
    with_meas_errs=False,
)

pyrometer_sensor = sens.SensorLibrary.pyrometer(
    sim_data=sim_data,
    sensor_position=scanner_pos,
    aim_direction=beam_dir,
    temp_key="temperature",
    spatial_dims=sens.EDim.TWOD,
    with_meas_errs=False,
)

meas_lidar = lidar_sensor.sim_measurements()
meas_pyro = pyrometer_sensor.sim_measurements()
times = sim_data.time

# %%
# 3. Visualise standoff distance and surface temperature
# ------------------------------------------------------
show_plots: bool = False
output_path = Path("pyvale-output/extsensorsim")
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, dpi=100)

ax1.plot(
    times,
    meas_lidar[0, 0, :],
    "C0-",
    linewidth=2.2,
    label="LIDAR Standoff Distance",
)
ax1.set_ylabel("Distance (mm)", fontsize=11)
ax1.set_title("Non-Contact Optical Transducers", fontsize=12)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(frameon=True, facecolor="white", edgecolor="none")

ax2.plot(
    times,
    meas_pyro[0, 0, :],
    "C3-",
    linewidth=2.2,
    label="Optical Pyrometer Surface Temp",
)
ax2.set_xlabel("Time (s)", fontsize=11)
ax2.set_ylabel("Temperature (°C)", fontsize=11)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(frameon=True, facecolor="white", edgecolor="none")

fig.tight_layout()

save_fig = output_path / "ext_ex8e_ray_sensors.png"
fig.savefig(save_fig, dpi=200, bbox_inches="tight")
print(f"Saved plot to: {save_fig}")

# %%
# .. image:: ../../../../_static/ext_ex8e_ray_sensors.png
#    :alt: Ray-Casting Optical LIDAR and Pyrometer Transducers
#    :width: 700px
#    :align: center

if show_plots:
    plt.show()
else:
    plt.close(fig)
