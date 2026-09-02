# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Error Graph: Complex Environmental and Transducer Network
================================================================================

In this example, we construct a comprehensive, multi-stage transducer
measurement error network combining:
1. Physical field spatial uncertainty (`ErrSysField`).
2. Transducer non-linear calibration curve with exact Newton-Raphson inversion
   (`ErrSysCalibration`).
3. Multiplicative gain perturbation (`EErrOp.MULTIPLY`).
4. Additive Gaussian white noise and static electronics offset (`ErrRandGen`,
   `ErrSysOffset`).
5. Convergence and multi-sensor inspection.

This showcases the full flexibility of Pyvale's Directed Acyclic Graph (DAG)
machinery for modeling realistic experimental instrumentation chains.
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
# We place 4 sensors across the domain
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
# 3. Define Error Models
# ----------------------

# 3.1 Spatial uncertainty (position jitter)
pos_gen = sens.GenNormal(std=0.5)
field_err = sens.ErrSysField(
    field=sens_array.get_field(),
    field_err_data=sens.ErrFieldData(pos_rand_xyz=(pos_gen, pos_gen, None)),
    err_dep=sens.EErrDep.DEPENDENT,
)

# 3.2 Non-linear calibration with Newton-Raphson inversion
def calib_assumed(v: np.ndarray) -> np.ndarray:
    return 24.3 * v + 0.616

def calib_truth(v: np.ndarray) -> np.ndarray:
    return -0.019 + 25.4 * v - 0.42 * v**2 + 0.04 * v**3

def calib_truth_prime(v: np.ndarray) -> np.ndarray:
    return 25.4 - 0.84 * v + 0.12 * v**2

calib_err = sens.ErrSysCalibration(
    assumed_calib=calib_assumed,
    truth_calib=calib_truth,
    truth_calib_prime=calib_truth_prime,
    cal_range=(0.0, 10.0),
    use_newton=True,
    err_dep=sens.EErrDep.DEPENDENT,
)

# 3.3 Multiplicative gain error (e.g. 2% amplifier sensitivity tolerance)
class ErrSysGain(sens.IErrSimulator):
    """Simple systematic scaling / gain error."""

    __slots__ = ("_gain", "_dep")

    def __init__(self, gain: float, dep: sens.EErrDep) -> None:
        self._gain = gain
        self._dep = dep

    def get_error_dep(self) -> sens.EErrDep:
        return self._dep

    def set_error_dep(self, dep: sens.EErrDep) -> None:
        self._dep = dep

    def get_error_type(self) -> sens.EErrType:
        return sens.EErrType.SYSTEMATIC

    def reseed(self, seed: int | None = None) -> None:
        pass

    def sim_errs(
        self, err_basis: np.ndarray, sens_data: sens.SensorData
    ) -> tuple[np.ndarray, sens.SensorData]:
        err_out = (self._gain - 1.0) * err_basis
        return err_out, sens_data

gain_err = ErrSysGain(gain=1.02, dep=sens.EErrDep.DEPENDENT)

# 3.4 High-frequency Gaussian noise & systematic offset
white_noise = sens.ErrRandGen(
    generator=sens.GenNormal(std=0.2), err_dep=sens.EErrDep.INDEPENDENT
)
elec_offset = sens.ErrSysOffset(offset=0.5, err_dep=sens.EErrDep.INDEPENDENT)

#%%
# 4. Construct Complex DAG Pipeline
# ---------------------------------
# Topology:
#
#                    (Ground Truth)
#                         |
#                  [spatial_jitter]
#                         |
#                    [calib_nonlin]
#                         |
#                     [amp_gain]
#                    /          \
#                   /            \
#         [white_noise]        [elec_offset]
#                   \            /
#                    \          /
#                 (Merged Leaves)

builder = sens.ErrGraphBuilder()

builder.add_root(
    name="spatial_jitter",
    simulator=field_err,
    op=sens.EErrOp.ADD,
)

builder.add_child(
    name="calib_nonlin",
    simulator=calib_err,
    parent="spatial_jitter",
    op=sens.EErrOp.ADD,
)

builder.add_child(
    name="amp_gain",
    simulator=gain_err,
    parent="calib_nonlin",
    op=sens.EErrOp.ADD,
)

builder.add_child(
    name="white_noise",
    simulator=white_noise,
    parent="amp_gain",
    op=sens.EErrOp.ADD,
)

builder.add_child(
    name="elec_offset",
    simulator=elec_offset,
    parent="amp_gain",
    op=sens.EErrOp.ADD,
)

graph = builder.build(
    meas_shape=sens_array.get_measurement_shape(),
    sensor_data_initial=sens_data,
    opts=sens.ErrGraphOpts(store_node_outputs=True),
)
sens_array.set_error_graph(graph)

#%%
# 5. Simulate Measurements & Output Analytics
# -------------------------------------------
measurements = sens_array.sim_measurements()

print(80 * "=")
print("Complex Environmental Transducer Network Simulation:")
print(f"  Total graph nodes: {len(graph.get_nodes())}")
print(f"  Execution order:   {graph.get_execution_order()}")
print(80 * "=")

sens.print_measurements(
    sens_array=sens_array,
    sensors=0,
    components=0,
    time_steps=slice(-5, None),
)
print(80 * "=")

#%%
# 6. Visualise Results
# --------------------
output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

fig, ax = sens.plot_time_traces(sens_array, comp_key="temperature")
save_traces = output_path / "ext_ex6d_complex_network_traces.png"
fig.savefig(save_traces, dpi=200, bbox_inches="tight")
print(f"Saved trace plot to: {save_traces}")
