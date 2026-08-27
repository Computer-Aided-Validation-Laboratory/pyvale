# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Load Cells and Surface Traction Sensors
================================================================================

In experimental structural testing, load cells and multi-axis force washers
measure the total resultant force vector :math:`\\mathbf{F}` acting across a
contact interface :math:`\\Gamma_c`:
.. math::
    \\mathbf{F} = \\iint_{\\Gamma_c}
      \\mathbf{t}(\\mathbf{x}, t) \\, \\mathrm{d}A
    = \\iint_{\\Gamma_c} \\boldsymbol{\\sigma}(\\mathbf{x}, t) \\cdot
      \\mathbf{n} \\, \\mathrm{d}A

In this example, we demonstrate:
1. Assembling a multi-axis load cell using `SensorLibrary.load_cell()`.
2. Integrating 3D surface traction fields in `EIntegrationMode.ACCUMULATE` mode.
3. Comparing measured normal force with the applied reaction loads.
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
# 1. Load 2D/3D mechanical simulation data
# ----------------------------------------
data_path: Path = dataset.mechanical_2d_path()
sim_data: io.SimData = mh.ExodusLoader(data_path).load_all_sim_data()
sim_data = sens.scale_length_units(
    scale=1000.0, sim_data=sim_data, disp_keys=None
)

# %%
# 2. Deploy multi-axis load cells across contact boundaries
# ---------------------------------------------------------
# Load Cell A: Mounted at top boundary (y = 50 mm), normal pointing in +Y
# Load Cell B: Mounted at right boundary (x = 100 mm), normal pointing in +X
mount_a = np.array([25.0, 50.0, 0.0])
mount_b = np.array([100.0, 25.0, 0.0])

load_cell_a = sens.SensorLibrary.load_cell(
    sim_data=sim_data,
    mount_position=mount_a,
    contact_area_x=10.0,
    contact_area_y=1.0,
    normal=(0.0, 1.0, 0.0),
    spatial_dims=sens.EDim.TWOD,
    with_meas_errs=False,
)

load_cell_b = sens.SensorLibrary.load_cell(
    sim_data=sim_data,
    mount_position=mount_b,
    contact_area_x=1.0,
    contact_area_y=10.0,
    normal=(1.0, 0.0, 0.0),
    spatial_dims=sens.EDim.TWOD,
    with_meas_errs=False,
)

meas_a = load_cell_a.sim_measurements()
meas_b = load_cell_b.sim_measurements()
times = sim_data.time

# %%
# 3. Visualise contact force evolution
# ------------------------------------
show_plots: bool = False
output_path = Path("pyvale-output/extsensorsim")
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
ax.plot(
    times,
    meas_a[0, 1, :],
    "C0-",
    linewidth=2.2,
    label="Load Cell A: Normal Force $F_y$ (N)",
)
ax.plot(
    times,
    meas_a[0, 0, :],
    "C0--",
    linewidth=1.8,
    label="Load Cell A: Shear Force $F_x$ (N)",
)
ax.plot(
    times,
    meas_b[0, 0, :],
    "C3-.",
    linewidth=2.2,
    label="Load Cell B: Normal Force $F_x$ (N)",
)

ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel("Resultant Contact Force (N)", fontsize=11)
ax.set_title("Multi-Axis Load Cell Contact Forces", fontsize=12)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(frameon=True, facecolor="white", edgecolor="none")
fig.tight_layout()

save_fig = output_path / "ext_ex8b_load_cell.png"
fig.savefig(save_fig, dpi=200, bbox_inches="tight")
print(f"Saved plot to: {save_fig}")

# %%
# .. image:: ../../../../_static/ext_ex8b_load_cell.png
#    :alt: Multi-Axis Load Cell and Surface Traction Measurements
#    :width: 700px
#    :align: center

if show_plots:
    plt.show()
else:
    plt.close(fig)
