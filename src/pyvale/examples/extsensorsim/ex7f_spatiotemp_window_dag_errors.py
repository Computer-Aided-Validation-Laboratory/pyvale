# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
DAG Errors: Spatio-Temporal Windowing in Directed Acyclic Graphs
================================================================================

In complex experimental systems, spatio-temporal windowing operates as the
underlying physical integration mechanism, while calibration drifts, positioning
uncertainties, and electronic noise operate concurrently in a Directed Acyclic
Graph (DAG).

In this example, we demonstrate:
1. Combining `SensorsSpatial` with an `ErrGraph` pipeline.
2. Modeling spatial averaging across a rectangular gauge area while simulating
   independent branches for spatial positioning jitter and thermal drift.
3. Quantifying total sensor uncertainty combining physical integration and
   transducer errors.
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
# 2. Configure spatial sensor array
# ---------------------------------
sens_pos = np.array([
    [10.0, 10.0, 0.0],
    [15.0, 15.0, 0.0],
    [20.0, 20.0, 0.0],
])
sample_times = sim_data.time

sens_data = sens.SensorData(positions=sens_pos, sample_times=sample_times)
field = sens.FieldScalar(
    sim_data=sim_data,
    comp_key="temperature",
    spatial_dims=sens.EDim.TWOD,
)

# 4 mm x 4 mm active square gauge area
gauge_window = sens.SpatialWindowRectangle(
    length_x=4.0,
    length_y=4.0,
    integ_rule=sens.IntegrationGaussLegendre(order=2),
)

sensors = sens.SensorsSpatial(
    sensor_data=sens_data,
    field=field,
    spatial_window=gauge_window,
)

# %%
# 3. Build DAG error pipeline
# ---------------------------
# Branch A: Spatial positioning noise (simulated calibration/mounting error)
err_pos = sens.ErrSysGen(
    generator=sens.GenNormal(std=0.5, mean=0.0),
)

# Branch B: Instrumentation thermal drift
err_drift = sens.ErrSysGen(
    generator=sens.GenNormal(std=0.2, mean=0.5),
)

# Branch C: Random measurement noise
err_noise = sens.ErrRandGen(
    generator=sens.GenUniform(low=-0.3, high=0.3),
)

builder = sens.ErrGraphBuilder()
builder.add_root(
    name="pos_err",
    simulator=err_pos,
    op=sens.EErrOp.ADD,
)
builder.add_root(
    name="drift",
    simulator=err_drift,
    op=sens.EErrOp.ADD,
)
builder.add_child(
    name="noise",
    simulator=err_noise,
    parent="drift",
    op=sens.EErrOp.ADD,
)

graph = builder.build(
    meas_shape=sensors.get_measurement_shape(),
    sensor_data_initial=sens_data,
    opts=sens.ErrGraphOpts(store_node_outputs=True),
)

sensors.set_error_graph(graph)

# %%
# 4. Simulate measurements and extract error components
# -----------------------------------------------------
measurements = sensors.sim_measurements()
truth = sensors.get_truth()
tot_errors = sensors.get_errors_total()

times = sensors.get_sample_times()

# %%
# 5. Plot simulated signals with error envelopes
# ----------------------------------------------
show_plots: bool = False

output_path = Path.cwd() / "pyvale-output" / "extsensorsim"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

for ss in range(sens_pos.shape[0]):
    ax.plot(
        times,
        truth[ss, 0, :],
        color=f"C{ss}",
        linestyle="--",
        label=f"Truth Sensor {ss+1}",
    )
    ax.plot(
        times,
        measurements[ss, 0, :],
        color=f"C{ss}",
        linestyle="-",
        marker="o",
        markersize=3,
        label=f"Meas Sensor {ss+1}",
    )

ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel("Temperature (°C)", fontsize=11)
ax.set_title("Spatial Window Sensor with DAG Error Pipeline", fontsize=12)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(frameon=True, facecolor="white", ncol=2)
fig.tight_layout()

save_fig = output_path / "ext_ex7f_spatiotemp_dag.png"
fig.savefig(save_fig, dpi=200, bbox_inches="tight")
print(f"Saved plot to: {save_fig}")

# %%
# .. image:: ../../../../_static/ext_ex7f_spatiotemp_dag.png
#    :alt: Spatial window sensor with DAG error pipeline
#    :width: 700px
#    :align: center

if show_plots:
    plt.show()
else:
    plt.close(fig)
