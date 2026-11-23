# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Simulation sweep sampling
================================================================================
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
figure_numbers = count(1)

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
for nn, pp in parameter_sets.items():
    solve_result: mh.RunResult = runner.run(
        input_path,
        input_params=pp,
        output_dir=run_root / nn,
    )

    sims[nn] = mh.ExodusLoader(solve_result.output_file).load_all_sim_data()

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

sens.scale_length_units(
    1000.0, list(sims.values()), disp_keys=("disp_x", "disp_y", "disp_z")
)

sens_times = np.linspace(nominal_sim.time[0], nominal_sim.time[-1], 5)
strain_norm_keys = ("strain_xx", "strain_yy", "strain_zz")
strain_dev_keys = ("strain_xy", "strain_xz", "strain_yz")
strain_keys = (*strain_norm_keys, *strain_dev_keys)

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
        units="-",
        tag="SG",
        components=("xx", "yy", "zz", "xy", "xz", "yz"),
    ),
)


err_chain = [sens.ErrRandGen(sens.GenNormal(std=1.0e-7, seed=42))]
strain_sensors.set_error_chain(err_chain)


measurements = np.stack(
    [strain_sensors.sim_measurements() for _ in range(6)], axis=-1
)  # (6 points, 6 tensor components, 5 times, 6 aleatory repeats)

# %%
strain_tag = "cube_strain"
observables = {
    strain_tag: vl.ValidObservable(
        components=strain_keys,
        geometry=vl.EValidGeometry.POINT,
        coords=sens_coords,
        times=sens_times,
        phys_units="-",
        coord_units="mm",
        time_units="s",
    )
}
model: dict[str, vl.ValidArray] = vl.sample_sim_sweep_for_valid(
    {
        strain_tag: sens.FieldTensor(
            nominal_sim, strain_norm_keys, strain_dev_keys, sens.EDim.THREED
        )
    },
    observables,
    [((ii,), ss) for ii, ss in enumerate(sims.values())],
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
        kk: vl.ValidArray(
            np.array(
                [oo.get(kk, nn) for oo in parameter_sets.values()]
            ),
            parameter_axes,
        )
        for kk, nn in (("surfHeatFlux", 4.67e6), ("ThermExp", 17.8e-6))
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
for ii, kk in enumerate(strain_keys):
    ecdf_plot: vl.ECDFPlot = vl.plot_sim_exp_cdfs(
        valid_case,
        strain_tag,
        point_idxs=point_idxs,
        component_idx=ii,
        time_idx=4,
        valid_plot_opts=vl.ValidPlotOpts(x_label=f"{kk} [-]"),
    )
    ecdf_plot.figure.savefig(
        output_path / f"val_ex09_fig{next(figure_numbers):02d}_{kk}_cdfs.png",
        bbox_inches="tight",
    )
    
    fig, _ = vl.plot_metric_time_points(
        valid_result,
        strain_tag,
        vl.EMetricType.MAVM,
        point_idxs=point_idxs,
        selectors={
            vl.EValidResAxis.COMPONENT: ii,
            vl.EValidResAxis.SIM_TRANS: 0,
        },
        valid_plot_opts=vl.ValidPlotOpts(title=kk),
    )
    fig.savefig(
        output_path / f"val_ex09_fig{next(figure_numbers):02d}_{kk}_mavm_time.png",
        bbox_inches="tight",
    )
    
    if not SHOW_PLOTS:
        plt.close("all")

# %%
for ss, nn in enumerate(parameter_sets):
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
    fig.savefig(
        output_path / f"val_ex09_fig{next(figure_numbers):02d}_{nn}_avm_time.png",
        bbox_inches="tight",
    )
    
if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
