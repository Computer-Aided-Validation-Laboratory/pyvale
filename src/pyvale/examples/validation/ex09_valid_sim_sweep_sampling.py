# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Simulation sweep sampling
================================================================================
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.validation as vl
import pyvale.validation.validpreparation as validprep

matplotlib.use("Agg")

# %%
SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")
output_path = Path.cwd() / "pyvale-output" / "valid_ex09_sim_sweep_sampling"
output_path.mkdir(parents=True, exist_ok=True)

# %%
input_path: Path = dataset.element_case_input_path(dataset.EElemTest.HEX8)
runner = mh.MooseRunner(timeout=120.0)
run_root: Path = mh.get_next_run_path(output_path / "solves")
parameter_sets: dict[str, dict[str, mh.InputValues]] = {
    "nominal": {},
    "heat_flux_plus_10pct": {"surfHeatFlux": 1.1 * 4.67e6},
    "thermal_expansion_plus_10pct": {"ThermExp": 1.1 * 17.8e-6},
}
sims: dict[str, io.SimData] = {}
for name, parameters in parameter_sets.items():
    solve_result: mh.RunResult = runner.run(
        input_path,
        input_params=parameters,
        output_dir=run_root / name,
    )
    sims[name] = mh.ExodusLoader(solve_result.output_file).load_all_sim_data()
nominal_sim: io.SimData = sims["nominal"]
exp_sim: io.SimData = sims["heat_flux_plus_10pct"]

# %%
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
for sim_data in sims.values():
    sens.scale_length_units(1000.0, sim_data, disp_keys=("disp_x", "disp_y", "disp_z"))
sens_times = np.linspace(nominal_sim.time[0], nominal_sim.time[-1], 5)
norm_keys = ("strain_xx", "strain_yy", "strain_zz")
dev_keys = ("strain_xy", "strain_xz", "strain_yz")
component_keys = (*norm_keys, *dev_keys)
sensor_data = sens.SensorData(positions=sens_coords, sample_times=sens_times)
strain_sensors: sens.SensorsPoint = sens.SensorFactory.tensor_point(
    exp_sim,
    sensor_data,
    norm_comp_keys=norm_keys,
    dev_comp_keys=dev_keys,
    spatial_dims=sens.EDim.THREED,
    descriptor=sens.SensorDescriptor(
        name="Strain",
        symbol=r"\varepsilon",
        units="1",
        tag="SG",
        components=("xx", "yy", "zz", "xy", "xz", "yz"),
    ),
)
strain_sensors.set_error_chain(
    [
        sens.ErrRandGen(sens.GenNormal(std=1.0e-7, seed=42)),
    ]
)
measurements = np.stack(
    [strain_sensors.sim_measurements() for repeat in range(6)], axis=-1
)  # (6 points, 6 tensor components, 5 times, 6 aleatory repeats)

# %%
strain_tag = "cube_strain"
observables = {
    strain_tag: vl.ValidObservable(
        components=component_keys,
        geometry=vl.EValidGeometry.POINT,
        coords=sens_coords,
        times=sens_times,
        phys_units="1",
        coord_units="mm",
        time_units="s",
    )
}
model: dict[str, vl.ValidArray] = validprep.sample_sim_sweep(
    {strain_tag: sens.FieldTensor(nominal_sim, norm_keys, dev_keys, sens.EDim.THREED)},
    observables,
    [((index,), sim_data) for index, sim_data in enumerate(sims.values())],
    sample_axes=(vl.EValidAxis.EPISTEMIC_TRANS,),
    sample_shape=(3,),
)
exp = {
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
parameter_axes = (vl.EValidAxis.EPISTEMIC_TRANS,)
sim_run_ids = vl.ValidArray(np.arange(len(parameter_sets)), parameter_axes)
sim_inputs = vl.ValidSampleInputs(
    parameters={
        key: vl.ValidArray(
            np.array(
                [overrides.get(key, nominal) for overrides in parameter_sets.values()]
            ),
            parameter_axes,
        )
        for key, nominal in (("surfHeatFlux", 4.67e6), ("ThermExp", 17.8e-6))
    },
    run_ids=sim_run_ids,
)
valid_case = vl.ValidCase(observables, model, exp, sim_inputs=sim_inputs)
print("Transferable run IDs:", sim_run_ids.data, tuple(parameter_sets))

# %%
valid_result: vl.ValidResult = vl.ValidEngine(
    [vl.MetricAVM(), vl.MetricMAVM()]
).run(valid_case)

# %%
metric_field: vl.MetricField = valid_result.metrics[strain_tag][
    vl.EMetricType.MAVM
]
print(
    "Model axes/shape:",
    model[strain_tag].axes,
    model[strain_tag].data.shape,
)
print("Experiment shape:", measurements.shape)
print(
    "Result axes/shape:",
    metric_field.axes,
    metric_field.outputs[vl.EMetricOut.D_PLUS].shape,
)

# %%
point_idxs = tuple(range(len(sens_coords)))
for component_idx, component_key in enumerate(component_keys):
    ecdf_plot: vl.ECDFPlot = vl.plot_sim_exp_cdfs(
        valid_case,
        strain_tag,
        point_idxs=point_idxs,
        component_idx=component_idx,
        time_idx=4,
        valid_plot_opts=vl.ValidPlotOpts(x_label=f"{component_key} [-]"),
    )
    ecdf_plot.figure.savefig(
        output_path / f"{component_key}_cdfs.png", bbox_inches="tight"
    )
    
    fig, _ = vl.plot_metric_time_points(
        valid_result,
        strain_tag,
        vl.EMetricType.MAVM,
        point_idxs=point_idxs,
        selectors={
            vl.EValidResAxis.COMPONENT: component_idx,
            vl.EValidResAxis.SIM_TRANS: 0,
        },
        valid_plot_opts=vl.ValidPlotOpts(title=component_key),
    )
    fig.savefig(
        output_path / f"{component_key}_mavm_time.png", bbox_inches="tight"
    )
    
    if not SHOW_PLOTS:
        plt.close("all")

# %%
for scenario, scenario_name in enumerate(parameter_sets):
    fig, _ = vl.plot_metric_time_points(
        valid_result,
        strain_tag,
        vl.EMetricType.AVM,
        point_idxs=point_idxs,
        selectors={
            vl.EValidResAxis.COMPONENT: 0,
            vl.EValidResAxis.SIM_TRANS: scenario,
        },
        valid_plot_opts=vl.ValidPlotOpts(title=f"strain_xx: {scenario_name}"),
    )
    fig.savefig(output_path / f"{scenario_name}_avm_time.png", bbox_inches="tight")
    
if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
