# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Error Graph: Multi-Array Experiment Simulation & Monte Carlo Analysis
================================================================================

In this example, we combine Directed Acyclic Graph (DAG) error models with
Pyvale's `ExperimentSimulator` engine.

We configure:
1. A thermal sensor array with a multi-branch DAG (spatial jitter + noise).
2. A mechanical displacement sensor array with a separate DAG (misalignment +
   electronic offset + noise).
3. A Monte Carlo campaign of 50 virtual experiments run across simulations.
4. Statistical aggregation (mean, std dev, 25%/75% quartiles) using
   `calc_exp_sim_stats()`.
5. Visualisation of the statistical confidence bounds across the virtual sensor
   traces.
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
data_path: Path = dataset.thermomechanical_2d_path()
sim_data: io.SimData = mh.ExodusLoader(data_path).load_all_sim_data()
sim_data = sens.scale_length_units(
    scale=1000.0,
    sim_data=sim_data,
    disp_keys=("disp_x", "disp_y"),
)
sim_dict = {"plate_2d": sim_data}

#%%
# 2. Build virtual sensor arrays
# ------------------------------
sim_dims = sens.simtools.get_sim_dims(sim_data)
sens_pos = sens.gen_pos_grid_inside(
    num_sensors=(2, 2, 1),
    x_lims=sim_dims["x"],
    y_lims=sim_dims["y"],
    z_lims=(0.0, 0.0),
)
sample_times = np.linspace(0.0, float(np.max(sim_data.time)), 40)
sens_data = sens.SensorData(positions=sens_pos, sample_times=sample_times)

# 2.1 Thermal scalar array
temp_array: sens.SensorsPoint = sens.SensorFactory.scalar_point(
    sim_data=sim_data,
    sensor_data=sens_data,
    comp_key="temperature",
    spatial_dims=sens.EDim.TWOD,
    descriptor=sens.DescriptorFactory.temperature(),
)

# 2.2 Mechanical vector displacement array
disp_array: sens.SensorsPoint = sens.SensorFactory.vector_point(
    sim_data=sim_data,
    sensor_data=sens_data,
    comp_keys=("disp_x", "disp_y"),
    spatial_dims=sens.EDim.TWOD,
    descriptor=sens.DescriptorFactory.displacement(),
)

#%%
# 3. Construct DAG Error Graphs for Each Sensor Array
# ---------------------------------------------------

# 3.1 Thermal Array DAG: Spatial Jitter + Random Noise
temp_pos_gen = sens.GenNormal(std=0.5)
temp_field_err = sens.ErrSysField(
    field=temp_array.get_field(),
    field_err_data=sens.ErrFieldData(
        pos_rand_xyz=(temp_pos_gen, temp_pos_gen, None)
    ),
    err_dep=sens.EErrDep.DEPENDENT,
)
temp_noise_err = sens.ErrRandGen(
    generator=sens.GenNormal(std=0.15),
    err_dep=sens.EErrDep.INDEPENDENT,
)

temp_builder = sens.ErrGraphBuilder()
temp_builder.add_root(
    name="spatial_jitter",
    simulator=temp_field_err,
    op=sens.EErrOp.ADD,
)
temp_builder.add_root(
    name="random_noise",
    simulator=temp_noise_err,
    op=sens.EErrOp.ADD,
)
temp_graph = temp_builder.build(
    meas_shape=temp_array.get_measurement_shape(),
    sensor_data_initial=sens_data,
)
temp_array.set_error_graph(temp_graph)

# 3.2 Displacement Array DAG: Offset + Noise
disp_offset = sens.ErrSysOffset(
    offset=0.005,  # 5 um offset
    err_dep=sens.EErrDep.INDEPENDENT,
)
disp_noise = sens.ErrRandGen(
    generator=sens.GenNormal(std=0.002),  # 2 um noise
    err_dep=sens.EErrDep.INDEPENDENT,
)

disp_builder = sens.ErrGraphBuilder()
disp_builder.add_root(
    name="offset",
    simulator=disp_offset,
    op=sens.EErrOp.ADD,
)
disp_builder.add_child(
    name="noise",
    simulator=disp_noise,
    parent="offset",
    op=sens.EErrOp.ADD,
)
disp_graph = disp_builder.build(
    meas_shape=disp_array.get_measurement_shape(),
    sensor_data_initial=sens_data,
)
disp_array.set_error_graph(disp_graph)

#%%
# 4. Run Monte Carlo Experiment Simulation
# ----------------------------------------
sensor_dict = {
    "temperature": temp_array,
    "displacement": disp_array,
}

exp_opts = sens.ExpSimOpts(workers=1, para=sens.EExpSimPara.ALL)
exp_keys = sens.ExpSimSaveKeys()

exp_sim = sens.ExperimentSimulator(
    sim_dict=sim_dict,
    sensor_arrays=sensor_dict,
    exp_sim_opts=exp_opts,
    exp_save_keys=exp_keys,
)

num_monte_carlo = 50
print(f"Running {num_monte_carlo} Monte Carlo virtual experiments...")
exp_data = exp_sim.run_experiments(num_exp_per_sim=num_monte_carlo)

# Compute statistical distribution over all runs
stats_data = sens.calc_exp_sim_stats(exp_data)

print(80 * "=")
print("Monte Carlo DAG Experiment Simulation Summary:")
for key, stats in stats_data.items():
    print(f"  Key {key}: mean shape = {stats.mean.shape}")
print(80 * "=")

#%%
# 5. Visualise Confidence Intervals
# ---------------------------------
output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

temp_stats = stats_data[("plate_2d", "temperature", "meas")]
times = sample_times
mean_trace = temp_stats.mean[0, 0, :]
std_trace = temp_stats.std[0, 0, :]

fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
ax.plot(times, mean_trace, "b-", label="Mean Measurement")
ax.fill_between(
    times,
    mean_trace - 2.0 * std_trace,
    mean_trace + 2.0 * std_trace,
    color="b",
    alpha=0.25,
    label=r"$\pm 2\sigma$ Confidence Interval",
)
ax.set_xlabel("Time (s)")
ax.set_ylabel(r"Temperature ($^\circ$C)")
ax.set_title("Temperature Sensor 0 - Monte Carlo DAG Error Envelope")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend()

save_fig = output_path / "ext_ex6e_monte_carlo_dag_envelope.png"
fig.savefig(save_fig, dpi=200, bbox_inches="tight")
print(f"Saved Monte Carlo plot to: {save_fig}")
