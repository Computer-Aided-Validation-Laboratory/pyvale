# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Error Graph: Linear Chain Conversion and Graph Introspection
================================================================================

In this example, we show how to:
1. Automatically convert existing linear error chains into equivalent DAG error
   graphs using `sens.err_chain_to_graph()`.
2. Understand how `err_chain_to_graph()` automatically wires up independent vs.
   dependent error nodes.
3. Introspect the resulting DAG structure (node list, edge dependencies,
   and topological execution order).
4. Simulate measurements using the converted DAG.
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
data_path: Path = dataset.thermal_2d_path()
sim_data: io.SimData = mh.ExodusLoader(data_path).load_all_sim_data()
sim_data = sens.scale_length_units(
    scale=1000.0, sim_data=sim_data, disp_keys=None
)

#%%
# 2. Build virtual sensor array
# -----------------------------
sim_dims = sens.simtools.get_sim_dims(sim_data)
sens_pos = sens.gen_pos_grid_inside(
    num_sensors=(2, 3, 1),
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
# 3. Create a Classic Error Chain
# -------------------------------
# In standard pyvale workflows, error models are assembled as a list:
# - Error 0: Spatial field perturbation (dependent)
# - Error 1: Fixed calibration offset (independent)
# - Error 2: Gaussian measurement noise (independent)

pos_rand = sens.GenNormal(std=0.5)
field_err = sens.ErrSysField(
    field=sens_array.get_field(),
    field_err_data=sens.ErrFieldData(pos_rand_xyz=(pos_rand, pos_rand, None)),
    err_dep=sens.EErrDep.DEPENDENT,
)

offset_err = sens.ErrSysOffset(
    offset=1.2,
    err_dep=sens.EErrDep.INDEPENDENT,
)

noise_err = sens.ErrRandGen(
    generator=sens.GenNormal(std=0.3),
    err_dep=sens.EErrDep.INDEPENDENT,
)

legacy_chain = [field_err, offset_err, noise_err]

#%%
# 4. Convert Linear Chain to DAG
# ------------------------------
# `err_chain_to_graph()` automatically builds an `ErrGraph` from the list.

graph = sens.err_chain_to_graph(
    err_chain=legacy_chain,
    meas_shape=sens_array.get_measurement_shape(),
    sensor_data_initial=sens_data,
    opts=sens.ErrGraphOpts(store_node_outputs=True),
)

print(80 * "=")
print("Auto-Converted Error Graph Introspection:")
print(f"  Total nodes:          {len(graph.get_nodes())}")
print(f"  Execution order:      {graph.get_execution_order()}")
print(80 * "=")

for node_name, node in graph.get_nodes().items():
    sim_name = type(node.simulator).__name__ if node.simulator else "None"
    print(f"Node '{node_name}': inputs={node.inputs}, sim={sim_name}")
print(80 * "=")

#%%
# 5. Apply DAG to Sensor Array & Simulate
# ---------------------------------------
sens_array.set_error_graph(graph)
measurements = sens_array.sim_measurements()

print(f"\nSimulated measurement array shape: {measurements.shape}")
sens.print_measurements(
    sens_array=sens_array,
    sensors=0,
    components=0,
    time_steps=slice(-5, None),
)
print(80 * "=")

#%%
# 6. Visualise Output Traces
# --------------------------
output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

fig, ax = sens.plot_time_traces(sens_array, comp_key="temperature")
save_traces = output_path / "ext_ex6c_chain_conversion_traces.png"
fig.savefig(save_traces, dpi=200, bbox_inches="tight")
print(f"Saved trace plot to: {save_traces}")
