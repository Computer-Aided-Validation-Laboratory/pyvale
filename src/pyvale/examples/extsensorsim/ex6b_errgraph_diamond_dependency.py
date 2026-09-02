# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Error Graph: Diamond Dependency and Intermediate State Inspection
================================================================================

In this example, we demonstrate a 'diamond' DAG topology where an upstream
error node feeds into two separate downstream error paths that subsequently
re-converge into the final measurement.

We also demonstrate how Pyvale DAGs propagate sensor state (`SignalState`),
such that sensor position updates from an upstream `ErrSysField` node are
automatically carried through to all child branches.

Finally, we show how to introspect intermediate signal states from arbitrary
nodes in the DAG after a simulation run.
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
    num_sensors=(2, 2, 1),
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
# 3. Construct Diamond Error Graph
# --------------------------------
# Topology:
#
#               (Ground Truth)
#                      |
#               [field_perturb] (Root: spatial position perturbation)
#                 /         \
#                /           \
#        [calib_curve]     [sensor_noise] (Two parallel child branches)
#                \           /
#                 \         /
#               (Merged Terminal State)

# Upstream: Sensor spatial position perturbation
pos_gen = sens.GenNormal(std=0.8)
field_err = sens.ErrSysField(
    field=sens_array.get_field(),
    field_err_data=sens.ErrFieldData(pos_rand_xyz=(pos_gen, pos_gen, None)),
    err_dep=sens.EErrDep.DEPENDENT,
)

# Branch 1: Transducer non-linear calibration curve
def calib_assumed(v: np.ndarray) -> np.ndarray:
    return 25.0 * v

def calib_truth(v: np.ndarray) -> np.ndarray:
    return 24.5 * v + 0.05 * v**2

def calib_truth_prime(v: np.ndarray) -> np.ndarray:
    return 24.5 + 0.10 * v

calib_err = sens.ErrSysCalibration(
    assumed_calib=calib_assumed,
    truth_calib=calib_truth,
    truth_calib_prime=calib_truth_prime,
    cal_range=(0.0, 20.0),
    use_newton=True,
    err_dep=sens.EErrDep.DEPENDENT,
)

# Branch 2: Additive Gaussian ADC noise
noise_err = sens.ErrRandGen(
    generator=sens.GenNormal(std=0.25),
    err_dep=sens.EErrDep.INDEPENDENT,
)

builder = sens.ErrGraphBuilder()

# Upstream diamond peak connected to ground truth
builder.add_root(
    name="field_perturb",
    simulator=field_err,
    op=sens.EErrOp.ADD,
)

# Left diamond branch
builder.add_child(
    name="calib_curve",
    simulator=calib_err,
    parent="field_perturb",
    op=sens.EErrOp.ADD,
)

# Right diamond branch
builder.add_child(
    name="sensor_noise",
    simulator=noise_err,
    parent="field_perturb",
    op=sens.EErrOp.ADD,
)

graph = builder.build(
    meas_shape=sens_array.get_measurement_shape(),
    sensor_data_initial=sens_data,
    opts=sens.ErrGraphOpts(store_node_outputs=True),
)
sens_array.set_error_graph(graph)

#%%
# 4. Simulate Measurements and Inspect Intermediate Nodes
# -------------------------------------------------------
measurements = sens_array.sim_measurements()

print(80 * "=")
print("Diamond Error Graph Execution Analysis:")
print(80 * "=")
print(f"Topological Execution Order: {graph.get_execution_order()}")

# Retrieve intermediate signal states from the DAG
node_outputs = graph.get_node_outputs()
assert node_outputs is not None

field_state = node_outputs["field_perturb"]
calib_state = node_outputs["calib_curve"]
noise_state = node_outputs["sensor_noise"]

truth_array = sens_array.get_truth()
sensor_idx = 0
time_step = -1  # Last time step

val_truth = truth_array[sensor_idx, 0, time_step]
val_field = field_state.values[sensor_idx, 0, time_step]
val_calib = calib_state.values[sensor_idx, 0, time_step]
val_noise = noise_state.values[sensor_idx, 0, time_step]
val_final = measurements[sensor_idx, 0, time_step]

print(f"\nSignal Values at Sensor {sensor_idx} (Final Time Step):")
print(f"  1. Nominal Truth:        {val_truth:.3f} degC")
print(f"  2. After Field Perturb:  {val_field:.3f} degC")
print(f"  3. Left Branch (Calib):  {val_calib:.3f} degC")
print(f"  4. Right Branch (Noise): {val_noise:.3f} degC")
print(f"  5. Merged Final Output:  {val_final:.3f} degC")
print(80 * "=")

#%%
# 5. Visualise Final Traces
# -------------------------
output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

fig, ax = sens.plot_time_traces(sens_array, comp_key="temperature")
save_traces = output_path / "ext_ex6b_diamond_dag_traces.png"
fig.savefig(save_traces, dpi=200, bbox_inches="tight")
print(f"Saved trace plot to: {save_traces}")
