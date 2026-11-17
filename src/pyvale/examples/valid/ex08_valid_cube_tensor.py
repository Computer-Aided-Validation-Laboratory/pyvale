# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Validation of 3D cube tensor observations
================================================================================

Follow the six cube-face sensor locations used in extsensorsim/ex3f_tens3d.
Synthetic measurements provide a controlled numerical experiment for learning
three-dimensional validation and tensor-component comparisons.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.figure import Figure

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.valid.validpreparation as validprep
from pyvale import valid
from pyvale.valid import validplot, validvis

matplotlib.use("Agg")

# %%
# Save figures and print data structures; enable interactive windows below.
SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")
output_path = Path.cwd() / "pyvale-output" / "valid_ex8_cube_tensor"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Load and transform data
# -----------------------------------------
# The packaged HEX20 data describe a 10 mm cube. Exodus stores coordinates
# in metres and tensor strains as dimensionless components. Scale lengths only;
# strain must retain its physical values.
input_path: Path = dataset.element_case_output_path(dataset.EElemTest.HEX20)
nominal_sim: io.SimData = mh.ExodusLoader(input_path).load_all_sim_data()
exp_sim = nominal_sim

# %%
# Six strain rosettes sit at the centres of the xz, xy and yz face pairs.
# All tensors use the global Cartesian frame with aligned sensor axes. Model
# and experiment have the same measurement operator and component convention.
coords = np.array(
    [
        [5.0, 0.0, 5.0],
        [5.0, 10.0, 5.0],
        [5.0, 5.0, 0.0],
        [5.0, 5.0, 10.0],
        [0.0, 5.0, 5.0],
        [10.0, 5.0, 5.0],
    ]
)  # (6 points, 3 Cartesian coordinates), mm
sens.scale_length_units(1000.0, nominal_sim, disp_keys=("disp_x", "disp_y", "disp_z"))
assert nominal_sim.time is not None
assert nominal_sim.node_vars is not None
times = np.linspace(nominal_sim.time[0], nominal_sim.time[-1], 5)
norm_keys = ("strain_xx", "strain_yy", "strain_zz")
dev_keys = ("strain_xy", "strain_xz", "strain_yz")
component_keys = (*norm_keys, *dev_keys)
sensor_data = sens.SensorData(positions=coords, sample_times=times)
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
# Small additive strain noise keeps zero shear components meaningful too.
# The seed fixes this teaching example's repeat measurements for reproducibility.
strain_sensors.set_error_chain(
    [
        sens.ErrRandGen(sens.GenNormal(std=1.0e-5, seed=42)),
    ]
)
measurements = np.stack(
    [strain_sensors.sim_measurements() for repeat in range(6)], axis=-1
)  # (6 points, 6 tensor components, 5 times, 6 aleatory repeats)

# %%
# Build the shared observables and sample the model through the same point operator.
# IField component keys explicitly preserve xx, yy, zz, xy, xz, yz ordering.
strain_tag = "cube_strain"
observables = {
    strain_tag: valid.ValidObservable(
        components=component_keys,
        geometry=valid.EValidGeometry.POINT,
        coords=coords,
        times=times,
        phys_units="1",
        coord_units="mm",
        time_units="s",
    )
}
model: dict[str, valid.ValidArray] = validprep.sample_sim_sweep(
    {strain_tag: sens.FieldTensor(nominal_sim, norm_keys, dev_keys, sens.EDim.THREED)},
    observables,
    [((), nominal_sim)],
)
exp = {
    strain_tag: valid.ValidArray(
        measurements,
        (
            valid.EValidAxis.POINT,
            valid.EValidAxis.COMPONENT,
            valid.EValidAxis.TIME,
            valid.EValidAxis.ALEATORY,
        ),
    )
}
valid_case = valid.ValidCase(observables, model, exp)

# %%
# 2. Run validation analysis
# --------------------------
# The model is deterministic within each selected parameter scenario, while
# the experiment has six repeat observations. INCLUDE handles repeated supports.
valid_result: valid.ValidResult = valid.ValidEngine(
    [valid.MetricAVM(), valid.MetricMAVM()]
).run(valid_case)

# %%
# 3. Post-processing and inspect data structures
# ----------------------------------------------
# No tensor component is flattened into the observation axis. Result axes start
# (6 points, 6 components, 5 times), followed by retained parameter scenarios.
metric_field: valid.ValidMetricField = valid_result.metrics[strain_tag][
    valid.EMetricType.MAVM
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
    metric_field.outputs[valid.EMetricOutput.D_PLUS].shape,
)

# %%
# 4. Analysis and visualisation
# ------------------------------
# Show all six face sites for every tensor component. The ECDF envelope
# includes every retained epistemic parameter setting without pooling samples.
point_idxs = tuple(range(len(coords)))
for component_idx, component_key in enumerate(component_keys):
    fig: Figure
    axes: np.ndarray
    ecdf_plot: validplot.ValidECDFPlot = validplot.plot_sim_exp_cdfs(
        valid_case,
        strain_tag,
        point_idxs=point_idxs,
        component_idx=component_idx,
        time_idx=4,
        valid_plot_opts=validplot.ValidPlotOpts(x_label=f"{component_key} [1]"),
    )
    fig, axes = ecdf_plot.figure, ecdf_plot.axes
    fig.savefig(output_path / f"{component_key}_cdfs.png", bbox_inches="tight")
    for metric_output, direction in (
        (valid.EMetricOutput.D_PLUS, "d_plus"),
        (valid.EMetricOutput.D_MINUS, "d_minus"),
    ):
        fig, axes = validplot.plot_valid_metric_time_points(
            valid_result,
            strain_tag,
            valid.EMetricType.MAVM,
            point_idxs=point_idxs,
            selectors={valid.EValidResAxis.COMPONENT: component_idx},
            output=metric_output,
            valid_plot_opts=validplot.ValidPlotOpts(title=component_key),
        )
        fig.savefig(
            output_path / f"{component_key}_{direction}_time.png", bbox_inches="tight"
        )
    if not SHOW_PLOTS:
        plt.close("all")
# %%
# Overlay the selected strain_xx metric on several independent background fields.
# The two colourbars deliberately identify different quantities: model field
# values on the mesh, and validation discrepancy at the six original face sites.
# Every field is evaluated at the final simulation step, matching validation t.
background_time_idx = nominal_sim.time.size - 1
normal = np.stack(
    [nominal_sim.node_vars[key][:, background_time_idx] for key in norm_keys], axis=-1
)  # (n_nodes, 3), normal components xx, yy, zz
shear = np.stack(
    [nominal_sim.node_vars[key][:, background_time_idx] for key in dev_keys], axis=-1
)  # (n_nodes, 3), tensor shear components xy, xz, yz
strain_tensors: np.ndarray = io.assemble_strain_tensor(
    normal,
    shear,
    shear_convention=io.EStrainShear.TENSOR,
)  # (n_nodes, 3, 3), symmetric small-strain tensor in global Cartesian axes
# Equivalent strain is sqrt(2/3 epsilon_dev:epsilon_dev): volumetric strain is
# excluded. Principal strains are descending; index zero is the first principal.
equivalent_strain: np.ndarray = io.calc_equivalent_strain(strain_tensors)
first_principal: np.ndarray = io.calc_principal_strains(strain_tensors)[:, 0]
background_fields: dict[str, tuple[str | np.ndarray, sens.SensorDescriptor]] = {
    "temperature": (
        "temperature",
        sens.SensorDescriptor(name="Temperature", symbol="T", units="degC", tag="SG"),
    ),
    "strain_xx": (
        "strain_xx",
        sens.SensorDescriptor(
            name="Strain xx", symbol=r"\varepsilon_{xx}", units="1", tag="SG"
        ),
    ),
    "equivalent_strain": (
        equivalent_strain,
        sens.SensorDescriptor(
            name="Equivalent strain", symbol=r"\varepsilon_{eq}", units="1", tag="SG"
        ),
    ),
    "first_principal_strain": (
        first_principal,
        sens.SensorDescriptor(
            name="First principal strain", symbol=r"\varepsilon_1", units="1", tag="SG"
        ),
    ),
}
# Show all face tags through the translucent mesh, including the far faces.
for background_name, (background_field, descriptor) in background_fields.items():
    for metric_output, direction in (
        (valid.EMetricOutput.D_PLUS, "d_plus"),
        (valid.EMetricOutput.D_MINUS, "d_minus"),
    ):
        pv_plot: pv.Plotter = validvis.plot_valid_metric_on_sim(
            valid_result,
            strain_tag,
            valid.EMetricType.MAVM,
            selectors={
                valid.EValidResAxis.COMPONENT: 0,
                valid.EValidResAxis.TIME: 4,
            },
            output=metric_output,
            sim_data=nominal_sim,
            sim_field=background_field,
            sim_time_idx=background_time_idx,
            descriptor=descriptor,
            sim_coord_units="mm",
            vis_opts=validvis.create_valid_scene_opts(interactive=SHOW_PLOTS),
            metric_opts=validvis.ValidMetricVisOpts(
                mesh_opacity=0.4, labels_always_visible=True
            ),
        )
        pv_plot.screenshot(
            output_path / f"strain_xx_{direction}_on_{background_name}.png"
        )
        if SHOW_PLOTS:
            pv_plot.show()
        pv_plot.close()

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
