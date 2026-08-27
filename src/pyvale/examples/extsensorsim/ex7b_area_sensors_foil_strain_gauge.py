# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Area Sensors: Rectangular Foil Strain Gauge Rosette
================================================================================

In experimental stress analysis, electrical resistance strain gauges have a
finite rectangular active grid area ($L_x \times L_y$) that integrates normal
strain across the grid.

To determine the full 2D in-plane strain state at a surface point, three
gauges are arranged in a 0°/45°/90° rosette configuration.

In this example, we demonstrate:
1. Creating 2D area sensors using `SpatialWindowRectangle`.
2. Modeling active grid averaging with 2D Gauss-Legendre quadrature.
3. Specifying 3D orientations for the rosette branches (0°, 45°, 90°).
4. Resolving principal strains from the simulated rosette measurements.
"""

from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# pyvale imports
import pyvale.sensorsim as sens
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.data as dataset


# %%
# 1. Load physics simulation data
# -------------------------------
# We load a 2D mechanics simulation.
data_path: Path = dataset.mechanical_2d_path()
sim_data: io.SimData = mh.ExodusLoader(data_path).load_all_sim_data()
sim_data = sens.scale_length_units(
    scale=1000.0, sim_data=sim_data, disp_keys=("disp_x", "disp_y")
)

# %%
# 2. Configure 0°/45°/90° rectangular strain gauge rosette
# --------------------------------------------------------
# Position rosette near stress concentration (x=20 mm, y=25 mm)
center_pos = np.array([
    [20.0, 25.0, 0.0],
    [20.0, 25.0, 0.0],
    [20.0, 25.0, 0.0],
])

rosette_angles = (
    Rotation.from_euler("z", 0.0, degrees=True),
    Rotation.from_euler("z", 45.0, degrees=True),
    Rotation.from_euler("z", 90.0, degrees=True),
)

sens_data = sens.SensorData(
    positions=center_pos,
    sample_times=sim_data.time,
    angles=rosette_angles,
)

# 3 mm x 2 mm active foil grid
gauge_window = sens.SpatialWindowRectangle(
    length_x=3.0,
    length_y=2.0,
    integ_rule=sens.IntegrationGaussLegendre(order=3),
)

strain_field = sens.FieldTensor(
    sim_data=sim_data,
    norm_comp_keys=("strain_xx", "strain_yy"),
    dev_comp_keys=("strain_xy",),
    spatial_dims=sens.EDim.TWOD,
)

rosette = sens.SensorsSpatial(
    sensor_data=sens_data,
    field=strain_field,
    spatial_window=gauge_window,
    integration_mode=sens.EIntegrationMode.AVERAGE,
)

# %%
# 3. Simulate measurements
# ------------------------
# Measurement shape: (3 sensors, 3 tensor components, n_times)
# Component 0 is normal strain along the local gauge x-axis (e11)
meas = rosette.sim_measurements()

eps_0 = meas[0, 0, :]   # 0° gauge (eps_a)
eps_45 = meas[1, 0, :]  # 45° gauge (eps_b)
eps_90 = meas[2, 0, :]  # 90° gauge (eps_c)

# %%
# 4. Resolve in-plane principal strains
# -------------------------------------
# Standard rectangular rosette reduction equations:
# eps_p1,2 = (eps_a + eps_c)/2 +/- sqrt(((eps_a-eps_b)^2 + (eps_b-eps_c)^2)/2)
center_strain = 0.5 * (eps_0 + eps_90)
radius_strain = np.sqrt(0.5 * ((eps_0 - eps_45)**2 + (eps_45 - eps_90)**2))

eps_p1 = center_strain + radius_strain
eps_p2 = center_strain - radius_strain
gamma_max = 2.0 * radius_strain

times = rosette.get_sample_times()

# %%
# 5. Plot rosette signals and resolved principal strains
# ------------------------------------------------------
show_plots: bool = False

output_path = Path.cwd() / "pyvale-output" / "extsensorsim"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=100)

ax1.plot(times, eps_0 * 1e6, "C0-", label=r"$\varepsilon_a$ (0°)")
ax1.plot(times, eps_45 * 1e6, "C1--", label=r"$\varepsilon_b$ (45°)")
ax1.plot(times, eps_90 * 1e6, "C2-.", label=r"$\varepsilon_c$ (90°)")
ax1.set_xlabel("Time (s)", fontsize=11)
ax1.set_ylabel(r"Normal Strain ($\mu\varepsilon$)", fontsize=11)
ax1.set_title("Rosette Branch Measurements", fontsize=12)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(frameon=True)

ax2.plot(times, eps_p1 * 1e6, "C3-", label=r"Principal $\varepsilon_1$")
ax2.plot(times, eps_p2 * 1e6, "C4--", label=r"Principal $\varepsilon_2$")
ax2.plot(times, gamma_max * 1e6, "k:", label=r"Max Shear $\gamma_{\max}$")
ax2.set_xlabel("Time (s)", fontsize=11)
ax2.set_ylabel(r"Strain ($\mu\varepsilon$)", fontsize=11)
ax2.set_title("Resolved Principal Strains", fontsize=12)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(frameon=True)

fig.tight_layout()

save_fig = output_path / "ext_ex7b_strain_rosette.png"
fig.savefig(save_fig, dpi=200, bbox_inches="tight")
print(f"Saved plot to: {save_fig}")

# %%
# 6. 3D Area Sensor Geometry Visualisation with PyVista
# -----------------------------------------------------
vis_opts = sens.VisOptsSimSensors(interactive=show_plots)
geom_opts = sens.VisOptsSensorGeom(
    area_opacity=0.8,
    color_nominal="orange",
    show_wireframe_edges=True,
)
img_save = sens.VisOptsImageSave(
    path=output_path / "ext_ex7b_strain_gauge_3d_geom.png"
)
pv_plot = sens.plot_sensors_on_sim(
    sensor_array=rosette,
    component="strain_xx",
    vis_opts=vis_opts,
    geom_opts=geom_opts,
    image_save_opts=img_save,
)
if show_plots:
    pv_plot.show()
else:
    pv_plot.close()

# %%
# .. image:: ../../../../_static/ext_ex7b_strain_rosette.png
#    :alt: Strain gauge rosette simulated measurements
#    :width: 700px
#    :align: center

if show_plots:
    plt.show()
else:
    plt.close(fig)
