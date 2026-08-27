# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Differential Sensors: Clip-On Extensometers and Thermopiles
================================================================================

In mechanical materials characterization, clip-on extensometers measure the
relative displacement between two knife-edge contact lines separated by initial
gauge length :math:`L_0 = \\|\\mathbf{x}_B - \\mathbf{x}_A\\|`:
.. math::
    \\varepsilon = \\frac{(\\mathbf{u}_B - \\mathbf{u}_A) \\cdot
                   \\mathbf{e}_{AB}}{L_0}

Similarly, differential thermocouples (thermopiles) measure the relative
temperature difference :math:`\\Delta T = T_B - T_A`.

In this example, we demonstrate:
1. Composing two `ISensorArray` instances with `SensorsDifferential`.
2. Assembling a tensile knife-edge extensometer using
   `SensorLibrary.extensometer()`.
3. Verifying the measured strain against local foil strain gauge readings.
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
# 2. Construct clip-on extensometer and local foil strain gauge
# -------------------------------------------------------------
# Gauge length L0 = 25 mm along X axis
anchor_a = np.array([10.0, 25.0, 0.0])
anchor_b = np.array([35.0, 25.0, 0.0])

extensometer = sens.SensorLibrary.extensometer(
    sim_data=sim_data,
    anchor_a=anchor_a,
    anchor_b=anchor_b,
    disp_keys=("disp_x", "disp_y"),
    knife_edge_length=5.0,
    spatial_dims=sens.EDim.TWOD,
    with_meas_errs=False,
)

gauge_center = 0.5 * (anchor_a + anchor_b)
strain_gauge = sens.SensorLibrary.strain_gauge(
    sim_data=sim_data,
    positions=gauge_center.reshape(1, 3),
    grid_length_x=3.0,
    grid_length_y=2.0,
    spatial_dims=sens.EDim.TWOD,
    with_meas_errs=False,
)

meas_ext = extensometer.sim_measurements()
meas_sg = strain_gauge.sim_measurements()
times = sim_data.time

# %%
# 3. Visualise measured tensile engineering strain
# ------------------------------------------------
show_plots: bool = False
output_path = Path("pyvale-output/extsensorsim")
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
ax.plot(
    times,
    meas_ext[0, 0, :] * 1e6,
    "C0-",
    linewidth=2.2,
    label="Clip-On Extensometer (25 mm gauge length)",
)
ax.plot(
    times,
    meas_sg[0, 0, :] * 1e6,
    "C3--",
    linewidth=1.8,
    label="Foil Strain Gauge (3 mm grid)",
)

ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel("Engineering Strain (με)", fontsize=11)
ax.set_title("Clip-On Extensometer vs. Foil Strain Gauge", fontsize=12)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(frameon=True, facecolor="white", edgecolor="none")
fig.tight_layout()

save_fig = output_path / "ext_ex8d_differential.png"
fig.savefig(save_fig, dpi=200, bbox_inches="tight")
print(f"Saved plot to: {save_fig}")

# %%
# 5. 3D Differential Sensor Geometry Visualisation with PyVista
# -------------------------------------------------------------
vis_opts = sens.VisOptsSimSensors(interactive=show_plots)
geom_opts = sens.VisOptsSensorGeom(line_radius=0.5)
img_save = sens.VisOptsImageSave(
    path=output_path / "ext_ex8d_extensometer_3d_geom.png"
)
pv_plot = sens.plot_sensors_on_sim(
    sensor_array=extensometer,
    component="disp_x",
    vis_opts=vis_opts,
    geom_opts=geom_opts,
    image_save_opts=img_save,
)
if show_plots:
    pv_plot.show()
else:
    pv_plot.close()

# %%
# .. image:: ../../../../_static/ext_ex8d_differential.png
#    :alt: Differential Sensors and Clip-On Extensometers
#    :width: 700px
#    :align: center

if show_plots:
    plt.show()
else:
    plt.close(fig)
