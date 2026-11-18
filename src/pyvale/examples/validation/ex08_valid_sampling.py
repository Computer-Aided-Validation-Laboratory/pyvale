# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Sample 3D scalar/vector/tensor fields and assemble SensorSim acquisitions.

A deterministic sensor-frame bias grows with time and x. The generated
acquisitions use small, symmetric repeat offsets, so the systematic signal is
recognisable. No stored solver data or pyvale-data files are modified.
"""

from dataclasses import replace
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.figure import Figure
from scipy.spatial.transform import Rotation

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.validation as vl
from pyvale.validation import validpreparation as validprep

matplotlib.use("Agg")
SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")
output_path = Path.cwd() / "pyvale-output" / "valid_ex8_sampling"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Load a 10 mm cube and keep four footprint centres strictly inside it.
# Scaling changes lengths and displacements to mm, never dimensionless strain.
sim_data: io.SimData = mh.ExodusLoader(
    dataset.element_case_output_path(dataset.EElemTest.HEX20)
).load_all_sim_data()
sens.scale_length_units(1000.0, sim_data, disp_keys=("disp_x", "disp_y", "disp_z"))
assert sim_data.time is not None and sim_data.node_vars is not None
# An explicitly derived global is a nodal arithmetic mean, not a volume average.
# replace creates a local record; original field arrays are borrowed unchanged.
sim_times = sim_data.time
sim_data = replace(
    sim_data,
    glob_vars={
        **(sim_data.glob_vars or {}),
        "nodal_mean_temperature": np.mean(sim_data.node_vars["temperature"], axis=0),
    },
)
coords = np.array([[2.0, 2.0, 5.0], [8.0, 2.0, 5.0], [2.0, 8.0, 5.0], [8.0, 8.0, 5.0]])
times = np.linspace(sim_times[0], sim_times[-1], 5)
normal_keys = ("strain_xx", "strain_yy", "strain_zz")
shear_keys = ("strain_xy", "strain_xz", "strain_yz")
# Scalar averaging and vector/tensor rotation are selected by SensorData.
# Both sources below use the same 30-degree sensor frame and .4 mm footprint.
sensor_data = sens.SensorData(
    positions=coords,
    sample_times=times,
    angles=(Rotation.from_euler("z", 30.0, degrees=True),),
    spatial_averager=sens.EIntSpatialType.RECT4PT,
    spatial_dims=np.array([0.4, 0.4, 0.0]),
)
observables = {
    "temperature_average": vl.ValidObservable(
        components=("temperature",),
        geometry=vl.EValidGeometry.POINT,
        coords=coords,
        times=times,
        phys_units="degC",
        coord_units="mm",
        time_units="s",
    ),
    "displacement_local": vl.ValidObservable(
        components=("disp_x", "disp_y", "disp_z"),
        geometry=vl.EValidGeometry.POINT,
        coords=coords,
        times=times,
        phys_units="mm",
        coord_units="mm",
        time_units="s",
    ),
    "strain_local": vl.ValidObservable(
        components=(*normal_keys, *shear_keys),
        geometry=vl.EValidGeometry.POINT,
        coords=coords,
        times=times,
        phys_units="1",
        coord_units="mm",
        time_units="s",
    ),
    "mean_temperature": vl.ValidObservable(
        components=("nodal_mean_temperature",),
        geometry=vl.EValidGeometry.GLOBAL,
        times=times,
        phys_units="degC",
        time_units="s",
    ),
}
fields: dict[str, sens.IField] = {
    "temperature_average": sens.FieldScalar(sim_data, "temperature", sens.EDim.THREED),
    "displacement_local": sens.FieldVector(
        sim_data, ("disp_x", "disp_y", "disp_z"), sens.EDim.THREED
    ),
    "strain_local": sens.FieldTensor(
        sim_data, normal_keys, shear_keys, sens.EDim.THREED
    ),
}

# %%
# 2. A single bound IField can be sampled directly. Sweeps reuse and rebind the
# field objects in place, leaving them on the final run (one run here; ex09 has
# three transferable runs). Global observables need no field dictionary entry.
single: vl.ValidArray = validprep.sample_field_for_valid(
    fields["temperature_average"],
    observables["temperature_average"],
    sensor_data=sensor_data,
)
sim_arrays: dict[str, vl.ValidArray] = validprep.sample_sim_sweep(
    fields,
    observables,
    [((), sim_data)],
    sensor_data={tag: sensor_data for tag in fields},
)
global_samples: vl.ValidArray = validprep.sample_sim_globals(
    sim_data, observables["mean_temperature"]
)
np.testing.assert_allclose(single.data, sim_arrays["temperature_average"].data)
np.testing.assert_allclose(global_samples.data, sim_arrays["mean_temperature"].data)

# 3. Emulate SensorSim's (repeat, point, component, time) acquisition export.
# Here the measurement differences are prescribed: 1..2 degC at the final time,
# with symmetric .02 degC repeat offsets. Larger x means greater discrepancy.
x_fraction = (coords[:, 0] - np.min(coords[:, 0])) / np.ptp(coords[:, 0])
time_fraction = (times - times[0]) / (times[-1] - times[0])
drift = (1.0 + x_fraction[:, None, None]) * time_fraction[None, None, :]
repeat_offsets = np.array([-0.02, 0.0, 0.02])
export = {
    ("synthetic", "temperature", "meas"): np.stack(
        [single.data + drift + offset for offset in repeat_offsets]
    ),
    ("synthetic", "temperature", "sens_times"): times,
}
acquisitions: list[dict[str, io.ExpData]] = validprep.extract_exp_sim_data(
    export,
    sim_key="synthetic",
    selections={
        "temperature_average": validprep.SensorOutputSelection("temperature", 0, coords)
    },
)
cases: list[vl.ValidCase] = []
for repeat, acquisition in enumerate(acquisitions):
    exp_arrays: dict[str, vl.ValidArray] = validprep.assemble_exp_arrays(
        acquisition, {"temperature_average": observables["temperature_average"]}
    )
    # Multi-component acquisitions can already be supplied as ValidArray.
    # Their component order stays in the same rotated sensor frame as the sim.
    vector_bias = 0.002 * drift * np.array([1.0, -0.5, 0.25])[None, :, None]
    tensor_bias = (
        20.0e-6 * drift * np.array([1.0, -0.5, 0.25, 0.1, -0.1, 0.2])[None, :, None]
    )
    exp_arrays["displacement_local"] = vl.ValidArray(
        sim_arrays["displacement_local"].data
        + vector_bias
        + repeat_offsets[repeat] * 1.0e-4,
        sim_arrays["displacement_local"].axes,
    )
    exp_arrays["strain_local"] = vl.ValidArray(
        sim_arrays["strain_local"].data + tensor_bias + repeat_offsets[repeat] * 1.0e-6,
        sim_arrays["strain_local"].axes,
    )
    exp_arrays["mean_temperature"] = vl.ValidArray(
        global_samples.data + time_fraction[None, None, :] + repeat_offsets[repeat],
        global_samples.axes,
    )
    cases.append(vl.ValidCase(observables, sim_arrays, exp_arrays))
valid_case: vl.ValidCase = validprep.concatenate_valid_cases(*cases)
valid_result: vl.ValidResult = vl.ValidEngine(
    [vl.MetricAVM(), vl.MetricMAVM()]
).run(valid_case)
print("Sampled axes:", {tag: array.axes for tag, array in sim_arrays.items()})
print("Exp. repeat axis:", valid_case.exp["temperature_average"].axes)

# %%
# 4. Inspect temperature and tensor histories without flattening components.
fig: Figure
axes: np.ndarray
for tag, component in (
    ("temperature_average", 0),
    ("displacement_local", 0),
    ("strain_local", 0),
):
    ecdf_plot: vl.ECDFPlot = vl.plot_sim_exp_cdfs(
        valid_case, tag, point_idxs=None, component_idx=component, time_idx=4
    )
    ecdf_plot.figure.savefig(output_path / f"{tag}_cdfs.png", bbox_inches="tight")
    if not SHOW_PLOTS:
        plt.close(ecdf_plot.figure)
    for metric_output, direction in (
        (vl.EMetricOut.D_PLUS, "d_plus"),
        (vl.EMetricOut.D_MINUS, "d_minus"),
    ):
        fig, axes = vl.plot_metric_time_points(
            valid_result,
            tag,
            vl.EMetricType.MAVM,
            point_idxs=(0, 1, 2, 3),
            selectors={vl.EValidResAxis.COMPONENT: component},
            output=metric_output,
        )
        fig.savefig(output_path / f"{tag}_{direction}_time.png", bbox_inches="tight")
        if not SHOW_PLOTS:
            plt.close(fig)
# Time/component selectors absent from a stored field must be omitted. The scalar
# temperature field has only physical axes; its non-singleton time is selected.
np.testing.assert_allclose(
    valid_result.metrics["temperature_average"][vl.EMetricType.AVM].outputs[
        vl.EMetricOut.DISTANCE
    ][:, 0, -1],
    drift[:, 0, -1],
    atol=1.0e-10,
)
np.savetxt(
    output_path / "sampling_bias.csv",
    np.column_stack(
        (
            coords[:, 0],
            drift[:, 0, -1],
            valid_result.metrics["temperature_average"][vl.EMetricType.AVM].outputs[
                vl.EMetricOut.DISTANCE
            ][:, 0, -1],
        )
    ),
    delimiter=",",
    header="x_mm,imposed_bias,avm",
    comments="",
)

# Independent background colourbars may show a stored field key or a derived
# nodal scalar. Here equivalent strain is in the global solver frame; the metric
# concerns the local sensor xx component, so both labels identify their frame.
assert sim_data.node_vars is not None
background_index = sim_times.size - 1
normal = np.stack(
    [sim_data.node_vars[key][:, background_index] for key in normal_keys], axis=-1
)
shear = np.stack(
    [sim_data.node_vars[key][:, background_index] for key in shear_keys], axis=-1
)
tensor: np.ndarray = io.assemble_strain_tensor(
    normal, shear, shear_convention=io.EStrainShear.TENSOR
)
equivalent: np.ndarray = io.calc_equivalent_strain(tensor)
for background_name, background in (
    ("temperature", "temperature"),
    ("equivalent_strain", equivalent),
):
    for metric_output, direction in (
        (vl.EMetricOut.D_PLUS, "d_plus"),
        (vl.EMetricOut.D_MINUS, "d_minus"),
    ):
        plotter: pv.Plotter = vl.plot_metric_on_sim(
            valid_result,
            "strain_local",
            vl.EMetricType.MAVM,
            selectors={vl.EValidResAxis.COMPONENT: 0, vl.EValidResAxis.TIME: 4},
            output=metric_output,
            sim_data=sim_data,
            sim_field=background,
            sim_time_idx=background_index,
            sim_coord_units="mm",
            descriptor=sens.SensorDescriptor(name=background_name, tag="S"),
            metric_opts=vl.MetricVisOpts(labels_always_visible=True),
            vis_opts=vl.create_valid_scene_opts(interactive=SHOW_PLOTS),
        )
        plotter.screenshot(
            output_path / f"strain_local_{direction}_on_{background_name}.png"
        )
        if SHOW_PLOTS:
            plotter.show()
        plotter.close()
if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
