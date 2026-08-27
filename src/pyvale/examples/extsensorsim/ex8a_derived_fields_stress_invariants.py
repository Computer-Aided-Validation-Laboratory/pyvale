# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Derived Field Sensors: Stress Invariants and Yield Criteria
================================================================================

In experimental stress analysis, physical sensors or digital image correlation
often measure raw strain/stress tensor fields, from which engineers compute
derived stress invariants such as:
1. Von Mises equivalent stress :math:`\\sigma_{\\text{vM}}`
2. Ordered principal stresses :math:`\\sigma_1 \\ge \\sigma_2`
3. Maximum shear stress :math:`\\tau_{\\max} = (\\sigma_1 - \\sigma_2)/2`
4. Mean hydrostatic stress
   :math:`\\sigma_h = \\text{tr}(\\boldsymbol{\\sigma})/2`

In this example, we demonstrate:
1. Wrapping raw tensor fields in `FieldTransformed` with
   `FieldTransformVonMises`, `FieldTransformPrincipal`, and
   `FieldTransformHydrostatic`.
2. Sampling spatial integration arrays directly on derived scalar/vector fields.
3. Adding user-defined custom functionals via `FieldTransformCustom`.
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
# 1. Load 2D mechanical simulation data
# -------------------------------------
data_path: Path = dataset.mechanical_2d_path()
sim_data: io.SimData = mh.ExodusLoader(data_path).load_all_sim_data()
sim_data = sens.scale_length_units(
    scale=1000.0, sim_data=sim_data, disp_keys=None
)

# %%
# 2. Construct raw tensor field and derived invariant fields
# ----------------------------------------------------------
norm_keys = ("stress_xx", "stress_yy")
dev_keys = ("stress_xy",)

raw_stress_field = sens.FieldTensor(
    sim_data=sim_data,
    norm_comp_keys=norm_keys,
    dev_comp_keys=dev_keys,
    spatial_dims=sens.EDim.TWOD,
)

# Derived fields using composable field transforms
field_von_mises = sens.FieldTransformed(
    field=raw_stress_field,
    transform=sens.FieldTransformVonMises(),
)

field_principal = sens.FieldTransformed(
    field=raw_stress_field,
    transform=sens.FieldTransformPrincipal(return_max_shear=True),
)

field_hydrostatic = sens.FieldTransformed(
    field=raw_stress_field,
    transform=sens.FieldTransformHydrostatic(),
)

# %%
# 3. Deploy spatial sensor array to sample stress invariants
# ----------------------------------------------------------
probe_pos = np.array([[12.0, 15.0, 0.0]])
sens_data = sens.SensorData(
    positions=probe_pos, sample_times=sim_data.time
)

sensor_vm = sens.SensorsSpatial(
    sensor_data=sens_data,
    field=field_von_mises,
    spatial_window=sens.SpatialWindowRectangle(length_x=3.0, length_y=3.0),
    descriptor=sens.SensorDescriptor(
        name="Von Mises Probe", tag="VM", symbol="σ_vM", units="MPa"
    ),
)

sensor_pr = sens.SensorsSpatial(
    sensor_data=sens_data,
    field=field_principal,
    spatial_window=sens.SpatialWindowRectangle(length_x=3.0, length_y=3.0),
    descriptor=sens.SensorDescriptor(
        name="Principal Probe", tag="PRIN", symbol="σ", units="MPa"
    ),
)

sensor_hyd = sens.SensorsSpatial(
    sensor_data=sens_data,
    field=field_hydrostatic,
    spatial_window=sens.SpatialWindowRectangle(length_x=3.0, length_y=3.0),
    descriptor=sens.SensorDescriptor(
        name="Hydrostatic Probe", tag="HYD", symbol="σ_h", units="MPa"
    ),
)

meas_vm = sensor_vm.sim_measurements()
meas_pr = sensor_pr.sim_measurements()
meas_hyd = sensor_hyd.sim_measurements()

times = sim_data.time

# %%
# 4. Visualise stress invariant evolution
# ---------------------------------------
show_plots: bool = False
output_path = Path("pyvale-output/extsensorsim")
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
ax.plot(
    times,
    meas_vm[0, 0, :],
    "C0-",
    linewidth=2.2,
    label="Von Mises Equivalent Stress $\\sigma_{\\text{vM}}$",
)
ax.plot(
    times,
    meas_pr[0, 0, :],
    "C2--",
    linewidth=1.8,
    label="Major Principal $\\sigma_1$",
)
ax.plot(
    times,
    meas_pr[0, 1, :],
    "C3:",
    linewidth=1.8,
    label="Minor Principal $\\sigma_2$",
)
ax.plot(
    times,
    meas_pr[0, 2, :],
    "C4-.",
    linewidth=1.8,
    label="Max Shear $\\tau_{\\max}$",
)
ax.plot(
    times,
    meas_hyd[0, 0, :],
    "k-.",
    linewidth=1.5,
    label="Mean Hydrostatic $\\sigma_h$",
)

ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel("Stress (MPa)", fontsize=11)
ax.set_title("Derived Stress Invariants at Critical Notch", fontsize=12)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9)
fig.tight_layout()

save_fig = output_path / "ext_ex8a_derived_fields.png"
fig.savefig(save_fig, dpi=200, bbox_inches="tight")
print(f"Saved plot to: {save_fig}")

# %%
# .. image:: ../../../../_static/ext_ex8a_derived_fields.png
#    :alt: Derived Stress Invariants and Yield Criteria
#    :width: 700px
#    :align: center

if show_plots:
    plt.show()
else:
    plt.close(fig)
