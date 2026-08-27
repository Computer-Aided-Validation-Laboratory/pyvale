# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Error Graph: Parallel Branches
================================================================================

In this example, we introduce Pyvale's Directed Acyclic Graph (DAG) error
simulation architecture (`ErrGraph` and `ErrGraphBuilder`).

Traditional error chains execute error models in a strictly linear sequence.
However, real experimental measurement systems often have concurrent, parallel
error mechanisms. For example:
- Branch A: Sensor spatial positioning uncertainty (interpolating the physical
  field at slightly perturbed coordinates).
- Branch B: Instrument electronics drift and random thermal walk noise.

These two independent error branches both originate from the nominal physics
truth and combine at the data acquisition interface.

Here, we demonstrate how to build, configure, simulate, and visualise this
parallel-branch error graph using `ErrGraphBuilder`.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# pyvale imports
import pyvale.sensorsim as sens
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.data as dataset


#%%
# 1. Load physics simulation data
# -------------------------------
# We load a 2D transient thermal finite element simulation.
data_path: Path = dataset.thermal_2d_path()
sim_data: io.SimData = mh.ExodusLoader(data_path).load_all_sim_data()
sim_data = sens.scale_length_units(
    scale=1000.0, sim_data=sim_data, disp_keys=None
)

#%%
# 2. Build virtual sensor array
# -----------------------------
# We place a 3x2 grid of temperature point sensors inside the domain.
sim_dims = sens.simtools.get_sim_dims(sim_data)
sens_pos = sens.gen_pos_grid_inside(
    num_sensors=(3, 2, 1),
    x_lims=sim_dims["x"],
    y_lims=sim_dims["y"],
    z_lims=(0.0, 0.0),
)

sample_times = np.linspace(0.0, float(np.max(sim_data.time)), 50)
sens_data = sens.SensorData(positions=sens_pos, sample_times=sample_times)

sens_array: sens.SensorsPoint = sens.SensorFactory.scalar_point(
    sim_data=sim_data,
    sensor_data=sens_data,
    comp_key="temperature",
    spatial_dims=sens.EDim.TWOD,
    descriptor=sens.DescriptorFactory.temperature(),
)

#%%
# 3. Construct the Parallel Error DAG
# -----------------------------------
# We construct an error graph with two parallel branches from the ground truth:
#
#   (Ground Truth)
#     |--> [pos_jitter] (ErrSysField: spatial position uncertainty)
#     |--> [elec_offset] (ErrSysOffset: static amplifier offset)
#            |--> [elec_noise] (ErrRandGen: Gaussian ADC noise)
#
# Terminal leaves [pos_jitter] and [elec_noise] are automatically combined.

# 3.1 Create error simulator components
pos_rand = sens.GenNormal(std=0.5)  # +/- 0.5 mm position jitter
field_err = sens.ErrSysField(
    field=sens_array.get_field(),
    field_err_data=sens.ErrFieldData(
        pos_rand_xyz=(pos_rand, pos_rand, None),
    ),
    err_dep=sens.EErrDep.DEPENDENT,
)

elec_offset = sens.ErrSysOffset(
    offset=0.75,  # 0.75 degC static amplifier offset
    err_dep=sens.EErrDep.INDEPENDENT,
)

elec_noise = sens.ErrRandGen(
    generator=sens.GenNormal(std=0.2),  # 0.2 degC random noise
    err_dep=sens.EErrDep.DEPENDENT,
)

# 3.2 Build the DAG using ErrGraphBuilder
builder = sens.ErrGraphBuilder()

# Branch A: Root node connected to nominal ground truth
builder.add_root(
    name="pos_jitter",
    simulator=field_err,
    op=sens.EErrOp.ADD,
)

# Branch B: Root node connected to ground truth -> child noise node
builder.add_root(
    name="elec_offset",
    simulator=elec_offset,
    op=sens.EErrOp.ADD,
)
builder.add_child(
    name="elec_noise",
    simulator=elec_noise,
    parent="elec_offset",
    op=sens.EErrOp.ADD,
)

graph = builder.build(
    meas_shape=sens_array.get_measurement_shape(),
    sensor_data_initial=sens_data,
    opts=sens.ErrGraphOpts(store_node_outputs=True),
)

# Attach the error graph to the sensor array
sens_array.set_error_graph(graph)

#%%
# 4. Simulate Measurements and Inspect DAG Output
# -----------------------------------------------
measurements = sens_array.sim_measurements()

print(80 * "=")
print("Parallel Error Graph Simulation Completed:")
print(f"  Measurement array shape: {measurements.shape}")
print(f"  Graph node count: {len(graph.nodes)}")
print(f"  Execution order: {graph.execution_order}")
print(80 * "=")

# Print diagnostic breakdown for sensor 0
sens_print = 0
comp_print = 0
time_last = 5
time_print = slice(measurements.shape[2] - time_last, measurements.shape[2])

print(f"Last {time_last} measurements of Sensor {sens_print}:")
sens.print_measurements(sens_array, sens_print, comp_print, time_print)
print(80 * "=")

#%%
# 5. Visualise Simulated Traces
# -----------------------------
output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

fig, ax = sens.plot_time_traces(sens_array, comp_key="temperature")
save_traces = output_path / "ext_ex6a_parallel_dag_traces.png"
fig.savefig(save_traces, dpi=200, bbox_inches="tight")
print(f"Saved trace plot to: {save_traces}")
