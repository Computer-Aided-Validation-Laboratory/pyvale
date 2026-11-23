# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Simulation sampling at sensors
================================================================================
"""

from itertools import count
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.validation as vl

matplotlib.use("Agg")

SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")

output_path = Path.cwd() / "pyvale-output" / "valid_ex08_sim_sensor_sampling"
output_path.mkdir(parents=True, exist_ok=True)
figure_numbers = count(1)

# %%
sim_data: io.SimData = mh.ExodusLoader(
    dataset.element_case_output_path(dataset.EElemTest.HEX20)
).load_all_sim_data()

disp_keys = ("disp_x", "disp_y", "disp_z")
sens.scale_length_units(1000.0, sim_data, disp_keys=disp_keys)

temp_key = "temperature"
strain_norm_keys = ("strain_xx", "strain_yy", "strain_zz")
strain_shear_keys = ("strain_xy", "strain_xz", "strain_yz")
for kk in (*strain_norm_keys, *strain_shear_keys):
    sim_data.node_vars[kk] = sim_data.node_vars[kk] * 1.0e6

sim_times = sim_data.time
if sim_data.glob_vars is None:
    sim_data.glob_vars = {}

sim_data.glob_vars["nodal_mean_temperature"] = np.mean(
    sim_data.node_vars[temp_key], axis=0
)

sens_coords = np.array(
    [[2.0, 2.0, 5.0], [8.0, 2.0, 5.0], [2.0, 8.0, 5.0], [8.0, 8.0, 5.0]]
)
sens_times = np.linspace(sim_times[0], sim_times[-1], 5)

observables = {
    "temperature_average": vl.ValidObservable(
        components=(temp_key,),
        geometry=vl.EValidGeometry.POINT,
        coords=sens_coords,
        times=sens_times,
        phys_units="degC",
        coord_units="mm",
        time_units="s",
    ),
    "displacement_local": vl.ValidObservable(
        components=disp_keys,
        geometry=vl.EValidGeometry.POINT,
        coords=sens_coords,
        times=sens_times,
        phys_units="mm",
        coord_units="mm",
        time_units="s",
    ),
    "strain_local": vl.ValidObservable(
        components=(*strain_norm_keys, *strain_shear_keys),
        geometry=vl.EValidGeometry.POINT,
        coords=sens_coords,
        times=sens_times,
        phys_units="µε",
        coord_units="mm",
        time_units="s",
    ),
    "mean_temperature": vl.ValidObservable(
        components=("nodal_mean_temperature",),
        geometry=vl.EValidGeometry.GLOBAL,
        times=sens_times,
        phys_units="degC",
        time_units="s",
    ),
}


fields: dict[str, sens.IField] = {
    "temperature_average": sens.FieldScalar(
        sim_data, temp_key, sens.EDim.THREED
    ),
    "displacement_local": sens.FieldVector(
        sim_data, disp_keys, sens.EDim.THREED
    ),
    "strain_local": sens.FieldTensor(
        sim_data, strain_norm_keys, strain_shear_keys, sens.EDim.THREED
    ),
}

# %%
sensor_data = sens.SensorData(
    positions=sens_coords,
    sample_times=sens_times,
    angles=(Rotation.from_euler("z", 30.0, degrees=True),),
    spatial_averager=sens.EIntSpatialType.RECT4PT,
    spatial_dims=np.array([0.4, 0.4, 0.0]),
)


sim_ref_temp: vl.ValidArray = vl.sample_field_for_valid(
    fields["temperature_average"],
    observables["temperature_average"],
    sensor_data=sensor_data,
)

sim_sweep_samples: dict[str, vl.ValidArray] = vl.sample_sim_sweep_for_valid(
    fields,
    observables,
    [((), sim_data)],
    sensor_data={kk: sensor_data for kk in fields},
)

sim_nodal_mean_temp: vl.ValidArray = vl.sample_sim_globals(
    sim_data, observables["mean_temperature"]
)


# %%
sens_x = sens_coords[:, 0]
x_fraction = (sens_x - sens_x.min()) / np.ptp(sens_x)
time_fraction = (sens_times - sens_times[0]) / np.ptp(sens_times)
spatial_time_ramp = (1.0 + x_fraction[:, None, None]) * time_fraction

exp_temp_bias = spatial_time_ramp
exp_disp_bias = (
    0.002 * spatial_time_ramp * np.array([1.0, -0.5, 0.25])[None, :, None]
)
exp_strain_bias = (
    20.0
    * spatial_time_ramp
    * np.array([1.0, -0.5, 0.25, 0.1, -0.1, 0.2])[None, :, None]
)
exp_mean_temp_bias = time_fraction[None, None, :]

exp_temp_repeat_offsets = np.array([-0.02, 0.0, 0.02])
exp_disp_repeat_offsets = np.array([-2.0e-6, 0.0, 2.0e-6])
exp_strain_repeat_offsets = np.array([-0.02, 0.0, 0.02])

synth_exp_samples: dict[str, vl.ValidArray] = {}
for kk, ss, bb, rr in (
    ("temperature_average", sim_ref_temp, exp_temp_bias, exp_temp_repeat_offsets),
    (
        "displacement_local",
        sim_sweep_samples["displacement_local"],
        exp_disp_bias,
        exp_disp_repeat_offsets,
    ),
    (
        "strain_local",
        sim_sweep_samples["strain_local"],
        exp_strain_bias,
        exp_strain_repeat_offsets,
    ),
    (
        "mean_temperature",
        sim_nodal_mean_temp,
        exp_mean_temp_bias,
        exp_temp_repeat_offsets,
    ),
):
    synth_exp_samples[kk] = vl.ValidArray(
        (ss.data + bb)[..., None] + rr,
        (*ss.axes, vl.EValidAxis.ALEATORY),
    )

valid_case = vl.ValidCase(observables, sim_sweep_samples, synth_exp_samples)

valid_result: vl.ValidResult = vl.ValidEngine([vl.MetricAVM(), vl.MetricMAVM()]).run(
    valid_case
)

print(80 * "-")
print("Sampled axes:", {kk: aa.axes for kk, aa in sim_sweep_samples.items()})
print("Exp. repeat axis:", valid_case.exp["temperature_average"].axes)
print(80 * "-")

# %%
for kk, ii in (
    ("temperature_average", 0),
    ("displacement_local", 0),
    ("strain_local", 0),
):
    ecdf_plot: vl.ECDFPlot = vl.plot_sim_exp_cdfs(
        valid_case,
        kk,
        point_idxs=None,
        component_idx=ii,
        time_idx=4,
        valid_plot_opts=vl.ValidPlotOpts(x_axis_offset=kk != "strain_local"),
    )
    ecdf_plot.figure.savefig(
        output_path / f"val_ex08_fig{next(figure_numbers):02d}_{kk}_cdfs.png",
        bbox_inches="tight",
    )

    if not SHOW_PLOTS:
        plt.close(ecdf_plot.figure)

    fig, _ = vl.plot_metric_time_points(
        valid_result,
        kk,
        vl.EMetricType.MAVM,
        point_idxs=(0, 1, 2, 3),
        selectors={vl.EValidResAxis.COMPONENT: ii},
    )
    fig.savefig(
        output_path / f"val_ex08_fig{next(figure_numbers):02d}_{kk}_mavm_time.png",
        bbox_inches="tight",
    )

    if not SHOW_PLOTS:
        plt.close(fig)

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
