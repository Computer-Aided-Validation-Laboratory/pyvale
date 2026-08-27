# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Spatial Kernels: Optical Point Spread Functions (PSF) & Gaussian Weighting
================================================================================

Real optical and electromagnetic sensors (such as Laser Doppler Vibrometers LDV,
infrared pyrometers, and photodetectors) do not have uniform spatial
sensitivity across their active spot. The focal laser spot has a Gaussian
intensity distribution (Point Spread Function, PSF).

In this example, we demonstrate:
1. Creating non-uniform spatial weighting kernels (`SpatialKernelGaussian` and
   `SpatialKernelTriangular`).
2. Attaching spatial kernels to `SpatialWindowDisk` support windows.
3. Observing the spatial filtering effect across a steep thermal gradient.
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
# 2. Configure optical spot sensor with Gaussian and Triangular PSFs
# ------------------------------------------------------------------
# Sensor center located at thermal gradient peak
spot_pos = np.array([[12.0, 15.0, 0.0]])
sens_data = sens.SensorData(
    positions=spot_pos,
    sample_times=sim_data.time,
)

field = sens.FieldScalar(
    sim_data=sim_data,
    comp_key="temperature",
    spatial_dims=sens.EDim.TWOD,
)

spot_radius = 5.0  # 5 mm laser spot radius

# A: Uniform disk (flat sensitivity)
win_uniform = sens.SpatialWindowDisk(
    radius=spot_radius,
    kernel=sens.SpatialKernelUniform(),
)

# B: Gaussian laser spot (sigma = 2.0 mm)
win_gaussian = sens.SpatialWindowDisk(
    radius=spot_radius,
    kernel=sens.SpatialKernelGaussian(sigma=(2.0, 2.0)),
)

# C: Triangular (cone) sensitivity decay
win_triangular = sens.SpatialWindowDisk(
    radius=spot_radius,
    kernel=sens.SpatialKernelTriangular(radii=(spot_radius, spot_radius)),
)

sensor_uniform = sens.SensorsSpatial(
    sensor_data=sens_data, field=field, spatial_window=win_uniform
)
sensor_gaussian = sens.SensorsSpatial(
    sensor_data=sens_data, field=field, spatial_window=win_gaussian
)
sensor_triangular = sens.SensorsSpatial(
    sensor_data=sens_data, field=field, spatial_window=win_triangular
)

# %%
# 3. Simulate measurements
# ------------------------
meas_uni = sensor_uniform.sim_measurements()
meas_gauss = sensor_gaussian.sim_measurements()
meas_tri = sensor_triangular.sim_measurements()

times = sensor_uniform.get_sample_times()

# %%
# 4. Plot PSF kernel comparisons
# ------------------------------
show_plots: bool = False

output_path = Path.cwd() / "pyvale-output" / "extsensorsim"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

ax.plot(
    times,
    meas_uni[0, 0, :],
    "C0--",
    linewidth=2.0,
    label="Uniform Disk ($R=5$ mm)",
)
ax.plot(
    times,
    meas_gauss[0, 0, :],
    "C3-",
    linewidth=2.0,
    label=r"Gaussian Laser Spot ($\sigma=2$ mm)",
)
ax.plot(
    times,
    meas_tri[0, 0, :],
    "C2-.",
    linewidth=2.0,
    label=r"Triangular Cone ($R=5$ mm)",
)

ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel("Temperature (°C)", fontsize=11)
ax.set_title(
    "Optical Sensor: Effect of Point Spread Function (PSF) Kernels",
    fontsize=12,
)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(frameon=True, facecolor="white")
fig.tight_layout()

save_fig = output_path / "ext_ex7d_optical_psf.png"
fig.savefig(save_fig, dpi=200, bbox_inches="tight")
print(f"Saved plot to: {save_fig}")

# %%
# .. image:: ../../../../_static/ext_ex7d_optical_psf.png
#    :alt: Point spread function spatial sensitivity comparison
#    :width: 700px
#    :align: center

if show_plots:
    plt.show()
else:
    plt.close(fig)
