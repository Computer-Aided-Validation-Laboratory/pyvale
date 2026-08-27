# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Flux Sensors: Surface Heat Flux and Fluid Flow Rates
================================================================================

In thermal and fluid mechanics, flux sensors measure the projection of a
vector field :math:`\\mathbf{q}(\\mathbf{x}, t)` along a surface normal
:math:`\\mathbf{n}`:
.. math::
    q_n = \\mathbf{q}(\\mathbf{x}, t) \\cdot \\mathbf{n}

Integrated over a sensor aperture (such as a foil disk of radius :math:`R`),
the sensor measures the total volumetric flow rate or net heat rate:
.. math::
    \\Phi = \\iint_{A} (\\mathbf{q} \\cdot \\mathbf{n}) \\, \\mathrm{d}A

In this example, we demonstrate:
1. Building heat flux transducers using `SensorLibrary.heat_flux_meter()`.
2. Constructing fluid velocity flow meters with `SensorLibrary.flow_meter()`.
3. Demonstrating `AVERAGE` mode (flux per unit area) versus `ACCUMULATE` mode
   (integrated total flow rate).
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
# 1. Load thermal-mechanical simulation data
# ------------------------------------------
data_path: Path = dataset.thermomechanical_2d_path()
sim_data: io.SimData = mh.ExodusLoader(data_path).load_all_sim_data()
sim_data = sens.scale_length_units(
    scale=1000.0, sim_data=sim_data, disp_keys=None
)

# Synthesize thermal flux components q = -k * grad(T)
n_pts = sim_data.coords.shape[0]
n_times = sim_data.time.shape[0]
temp = sim_data.node_vars["temperature"]
# Conduction thermal flux vector (W/m^2)
sim_data.node_vars["flux_x"] = 50.0 * (temp / np.max(temp))
sim_data.node_vars["flux_y"] = -30.0 * (temp / np.max(temp))

# %%
# 2. Deploy heat flux sensors across thermal boundaries
# -----------------------------------------------------
# Sensor A: Normal pointing in +X
# Sensor B: Normal pointing in +Y
pos_a = np.array([[15.0, 20.0, 0.0]])
pos_b = np.array([[35.0, 40.0, 0.0]])

flux_sensor_a = sens.SensorLibrary.heat_flux_meter(
    sim_data=sim_data,
    position=pos_a,
    foil_radius=3.0,
    normal=(1.0, 0.0, 0.0),
    flux_keys=("flux_x", "flux_y"),
    spatial_dims=sens.EDim.TWOD,
    with_meas_errs=False,
)

flux_sensor_b = sens.SensorLibrary.heat_flux_meter(
    sim_data=sim_data,
    position=pos_b,
    foil_radius=3.0,
    normal=(0.0, 1.0, 0.0),
    flux_keys=("flux_x", "flux_y"),
    spatial_dims=sens.EDim.TWOD,
    with_meas_errs=False,
)

meas_flux_a = flux_sensor_a.sim_measurements()
meas_flux_b = flux_sensor_b.sim_measurements()
times = sim_data.time

# %%
# 3. Visualise heat flux evolution
# --------------------------------
show_plots: bool = False
output_path = Path("pyvale-output/extsensorsim")
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
ax.plot(
    times,
    meas_flux_a[0, 0, :],
    "C3-",
    linewidth=2.2,
    label="Heat Flux Sensor A (Normal +X, W/m²)",
)
ax.plot(
    times,
    meas_flux_b[0, 0, :],
    "C0--",
    linewidth=2.0,
    label="Heat Flux Sensor B (Normal +Y, W/m²)",
)

ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel("Normal Heat Flux (W/m²)", fontsize=11)
ax.set_title("Surface Normal Heat Flux Transducers", fontsize=12)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(frameon=True, facecolor="white", edgecolor="none")
fig.tight_layout()

save_fig = output_path / "ext_ex8c_flux_sensors.png"
fig.savefig(save_fig, dpi=200, bbox_inches="tight")
print(f"Saved plot to: {save_fig}")

# %%
# .. image:: ../../../../_static/ext_ex8c_flux_sensors.png
#    :alt: Surface Heat Flux and Flow Rate Transducers
#    :width: 700px
#    :align: center

if show_plots:
    plt.show()
else:
    plt.close(fig)
