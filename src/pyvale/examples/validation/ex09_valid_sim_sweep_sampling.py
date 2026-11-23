# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Simulation sweep sampling
================================================================================

Sample a MOOSE parameter sweep at virtual strain sensors and compare it with
synthetic measurements.
"""

from itertools import count
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.validation as vl

# %%
matplotlib.use("Agg")

SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")

output_path = Path.cwd() / "pyvale-output" / "valid_ex09_sim_sweep_sampling"
output_path.mkdir(parents=True, exist_ok=True)
fig_nums = count(1)

# %%
# 1. Run the epistemic sweep
# --------------------------
#
# The sweep varies heat flux and thermal expansion around the nominal model.
# Use the heat-flux variant as the source for the synthetic experiment, then
# add independent measurement noise.
input_path: Path = dataset.element_case_input_path(dataset.EElemTest.HEX8)
runner = mh.MooseRunner(timeout=120.0)
run_root: Path = mh.get_next_run_path(output_path / "solves")

nominal_heat_flux = 4.67e6
nominal_thermal_expansion = 17.8e-6
param_sets: dict[str, dict[str, mh.InputValues]] = {
    "nominal": {},
    "heat_flux_plus_10pct": {"surfHeatFlux": 1.1 * nominal_heat_flux},
    "thermal_expansion_plus_10pct": {"ThermExp": 1.1 * nominal_thermal_expansion},
}

sims: dict[str, io.SimData] = {}
for nn, pp in param_sets.items():
    solve_result: mh.RunResult = runner.run(
        input_path,
        input_params=pp,
        output_dir=run_root / nn,
    )

    sims[nn] = mh.ExodusLoader(solve_result.output_file).load_all_sim_data()

nominal_sim: io.SimData = sims["nominal"]
exp_sim: io.SimData = sims["heat_flux_plus_10pct"]

# %%
# 2. Generate strain measurements
# -------------------------------
#
# The six rows of coordinates give the x, y, and z positions of the sensors.
# The tensor sensor samples the six strain components at shared times. Stack
# noisy acquisitions on the final aleatory axis to obtain point, component,
# time, and repeat dimensions.
sens_coords = np.array(
    [
        [5.0, 0.0, 5.0],
        [5.0, 10.0, 5.0],
        [5.0, 5.0, 0.0],
        [5.0, 5.0, 10.0],
        [0.0, 5.0, 5.0],
        [10.0, 5.0, 5.0],
    ]
)  # (6 points, 3 Cartesian coordinates), mm

sens.scale_length_units(
    1000.0, list(sims.values()), disp_keys=("disp_x", "disp_y", "disp_z")
)

sens_times = np.linspace(nominal_sim.time[0], nominal_sim.time[-1], 5)
strain_norm_keys = ("strain_xx", "strain_yy", "strain_zz")
strain_dev_keys = ("strain_xy", "strain_xz", "strain_yz")
strain_keys = (*strain_norm_keys, *strain_dev_keys)
sens.scale_nodal_fields(1.0e3, list(sims.values()), field_keys=strain_keys)

sensor_data = sens.SensorData(positions=sens_coords, sample_times=sens_times)

strain_sensors: sens.SensorsPoint = sens.SensorFactory.tensor_point(
    exp_sim,
    sensor_data,
    norm_comp_keys=strain_norm_keys,
    dev_comp_keys=strain_dev_keys,
    spatial_dims=sens.EDim.THREED,
    descriptor=sens.SensorDescriptor(
        name="Strain",
        symbol=r"\varepsilon",
        units="mε",
        tag="SG",
        components=("xx", "yy", "zz", "xy", "xz", "yz"),
    ),
)


exp_strain_noise_sd = 0.005  # mε
err_chain = [sens.ErrRandGen(sens.GenNormal(std=exp_strain_noise_sd, seed=42))]
strain_sensors.set_error_chain(err_chain)


measurements = np.stack(
    [strain_sensors.sim_measurements() for _ in range(6)], axis=-1
)  # (6 points, 6 tensor components, 5 times, 6 aleatory repeats)

# %%
# 3. Sample each simulation at the same sensors
# ---------------------------------------------
#
# A flat list of runs fills the single transferable axis in list order. The
# resulting model adds that axis after point, component, and time. For a
# multi-axis sweep, pass indexed run pairs with an index tuple in sample-axis
# order.
strain_tag = "cube_strain"
observables = {
    strain_tag: vl.ValidObservable(
        components=strain_keys,
        geometry=vl.EValidGeometry.POINT,
        coords=sens_coords,
        times=sens_times,
        phys_units="mε",
        coord_units="mm",
        time_units="s",
    )
}

strain_field = sens.FieldTensor(
    nominal_sim, strain_norm_keys, strain_dev_keys, sens.EDim.THREED
)

model: dict[str, vl.ValidArray] = vl.sample_sim_sweep_for_valid(
    fields={strain_tag: strain_field},
    observables=observables,
    runs=list(sims.values()),
    sample_axes=(vl.EValidAxis.EPISTEMIC_TRANS,),
)

exp: dict[str, vl.ValidArray] = {
    strain_tag: vl.ValidArray(
        measurements,
        (
            vl.EValidAxis.POINT,
            vl.EValidAxis.COMPONENT,
            vl.EValidAxis.TIME,
            vl.EValidAxis.ALEATORY,
        ),
    )
}

# %%
# 4. Record sweep provenance
# --------------------------
#
# ``ValidSampleInputs`` records run IDs and input settings on the same
# transferable axis as the sampled model. Each run index identifies both its
# parameter values and its sensor predictions.
param_axes = (vl.EValidAxis.EPISTEMIC_TRANS,)
sim_run_ids = vl.ValidArray(np.arange(len(param_sets)), param_axes)

heat_flux_values = np.array(
    [pp.get("surfHeatFlux", nominal_heat_flux) for pp in param_sets.values()]
)
thermal_expansion_values = np.array(
    [pp.get("ThermExp", nominal_thermal_expansion) for pp in param_sets.values()]
)
sim_input_params = {
    "surfHeatFlux": vl.ValidArray(heat_flux_values, param_axes),
    "ThermExp": vl.ValidArray(thermal_expansion_values, param_axes),
}
sim_inputs = vl.ValidSampleInputs(
    input_params=sim_input_params,
    run_ids=sim_run_ids,
)

valid_case = vl.ValidCase(observables, model, exp, sim_inputs=sim_inputs)
print(80*"-")
print("Transferable run IDs:", sim_run_ids.data, tuple(param_sets))
print(80*"-")

# %%
# 5. Calculate metrics across sweep cases
# ---------------------------------------
#
# The simulation transferable axis remains in each metric field. Select a run
# by its index to plot a case-specific metric time history.
valid_result: vl.ValidResult = vl.ValidEngine(
    [vl.MetricAVM(), vl.MetricMAVM()]
).run(valid_case)

# %%
metric_field: vl.MetricField = valid_result.metrics[strain_tag][vl.EMetricType.MAVM]

print("Model axes/shape:",model[strain_tag].axes,model[strain_tag].data.shape)
print("Experiment shape:", measurements.shape)
print(
    "Result axes/shape:",
    metric_field.axes,
    metric_field.metric_vals[vl.EMetricOut.D_PLUS].shape,
)
print(80*"-")

# %%
# 6. Compare the simulation envelope
# ----------------------------------
#
# The ECDF envelope spans predictions from the three simulation runs at each
# sensor. The MAVM time panels use only the nominal simulation case.
point_idxs = tuple(range(len(sens_coords)))
sim_envelope_view = vl.ValidECDFView(
    trans_idxs=tuple(range(len(sims))),
    bounds=True,
)

for ii, kk in enumerate(strain_keys):
    ecdf_plot: vl.ECDFPlot = vl.plot_sim_exp_cdfs(
        valid_case,
        strain_tag,
        sim_view=sim_envelope_view,
        point_idxs=point_idxs,
        component_idx=ii,
        time_idx=4,
        valid_plot_opts=vl.ValidPlotOpts(
            sim_label="Sim Env", x_label=f"{kk} [mε]", x_axis_offset=False
        ),
    )

    ecdf_plot.figure.suptitle(
        "Simulation envelope: nominal, +10% heat flux, +10% thermal expansion"
    )

    fig_file = f"val_ex09_fig{next(fig_nums):02d}_{kk}_cdfs.png"
    ecdf_plot.figure.savefig(output_path / fig_file, bbox_inches="tight")
    
    fig, _ = vl.plot_metric_time_points(
        valid_result,
        strain_tag,
        vl.EMetricType.MAVM,
        point_idxs=point_idxs,
        selectors={
            vl.EValidResAxis.COMPONENT: ii,
            vl.EValidResAxis.SIM_TRANS: 0,
        },
        valid_plot_opts=vl.ValidPlotOpts(title=f"{kk}: nominal simulation"),
    )

    fig_file = f"val_ex09_fig{next(fig_nums):02d}_{kk}_mavm_time.png"
    fig.savefig(output_path / fig_file, bbox_inches="tight")
    
    if not SHOW_PLOTS:
        plt.close("all")

# %%
# 7. Compare case-specific AVM histories
# --------------------------------------
#
# Plot the strain xx AVM for each run using the same point and component
# selections. The title identifies the input change.
for ss, nn in enumerate(param_sets):
    fig, _ = vl.plot_metric_time_points(
        valid_result,
        strain_tag,
        vl.EMetricType.AVM,
        point_idxs=point_idxs,
        selectors={
            vl.EValidResAxis.COMPONENT: 0,
            vl.EValidResAxis.SIM_TRANS: ss,
        },
        valid_plot_opts=vl.ValidPlotOpts(title=f"strain_xx: {nn}"),
    )

    fig_file = f"val_ex09_fig{next(fig_nums):02d}_{nn}_avm_time.png"
    fig.savefig(output_path / fig_file, bbox_inches="tight")
    
if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
