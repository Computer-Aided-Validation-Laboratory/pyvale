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

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.valid as val
import pyvale.valid.converttovaldata as valconvert
from pyvale.valid import valplot, valvis

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
input_path = dataset.element_case_output_path(dataset.EElemTest.HEX20)
nominal_sim = mh.ExodusLoader(input_path).load_all_sim_data()
experimental_sim = nominal_sim

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
times = np.linspace(nominal_sim.time[0], nominal_sim.time[-1], 5)
norm_keys = ("strain_xx", "strain_yy", "strain_zz")
dev_keys = ("strain_xy", "strain_xz", "strain_yz")
component_keys = (*norm_keys, *dev_keys)
sensor_data = sens.SensorData(positions=coords, sample_times=times)
strain_sensors = sens.SensorFactory.tensor_point(
    experimental_sim,
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
# Build the shared domain and sample the model through the same point operator.
# IField component keys explicitly preserve xx, yy, zz, xy, xz, yz ordering.
strain_tag = "cube_strain"
domain = val.ValDomain(
    {
        strain_tag: val.ValObservable(
            geometry=val.ValGeometry(
                val.EValGeometry.POINT, coords, coord_units="mm", coord_frame="cube"
            ),
            operator=val.EValOperator.SAMPLE,
            components=component_keys,
            times=times,
            phys_units="1",
            time_units="s",
        )
    }
)
model = valconvert.convert_sim_to_val(
    {
        strain_tag: lambda simulation: sens.FieldTensor(
            simulation, norm_keys, dev_keys, sens.EDim.THREED
        )
    },
    domain,
    [((), nominal_sim)],
)
experiment = {
    strain_tag: val.ValArray(
        measurements,
        (
            val.EValAxis.POINT,
            val.EValAxis.COMPONENT,
            val.EValAxis.TIME,
            val.EValAxis.ALEATORY,
        ),
    )
}
val_case = val.ValCase(domain, model, experiment)

# %%
# 2. Run validation analysis
# --------------------------
# The model is deterministic within each selected parameter scenario, while
# the experiment has six repeat observations. INCLUDE handles repeated supports.
mavm = val.MetricMAVM(duplicates=val.EMAVMDuplicates.INCLUDE)
val_result = val.ValidationEngine([val.MetricAVM(), mavm]).run(val_case)

# %%
# 3. Post-processing and inspect data structures
# ----------------------------------------------
# No tensor component is flattened into the observation axis. Result axes start
# (6 points, 6 components, 5 times), followed by retained parameter scenarios.
metric_field = val_result.get(strain_tag, val.EMetricType.MAVM)
print(
    "Model axes/shape:",
    model[strain_tag].axes,
    model[strain_tag].data.shape,
)
print("Experiment shape:", measurements.shape)
print("Result axes/shape:", metric_field.axes, metric_field.metric.shape)

# %%
# 4. Analysis and visualisation
# ------------------------------
# Show all six face sites for every tensor component. The nominal model is
# selected explicitly in a parameter sweep, so these panels compare like cells.
point_idxs = tuple(range(len(coords)))
for component_idx, component_key in enumerate(component_keys):
    fig, axes = valplot.plot_val_cdfs_points(
        val_case,
        val_result,
        strain_tag,
        point_idxs=point_idxs,
        component_idx=component_idx,
        time_idx=4,
        metric=mavm,
        labels=valplot.ValPlotLabels(x_label=f"{component_key} [1]"),
    )
    fig.savefig(output_path / f"{component_key}_cdfs.png", bbox_inches="tight")
    fig, axes = valplot.plot_val_metric_time_points(
        val_result,
        strain_tag,
        val.EMetricType.MAVM,
        point_idxs=point_idxs,
        selectors={val.EValResAxis.COMPONENT: component_idx},
        part=val.EValMetricPart.D_MINUS,
        labels=valplot.ValPlotLabels(title=component_key),
    )
    fig.savefig(output_path / f"{component_key}_d_minus_time.png", bbox_inches="tight")
    if not SHOW_PLOTS:
        plt.close("all")
# %%
# Overlay the selected strain_xx metric on several independent background fields.
# The two colourbars deliberately identify different quantities: model field
# values on the mesh, and validation discrepancy at the six original face sites.
# Every field is evaluated at the final simulation step, matching validation t.
background_time_idx = len(nominal_sim.time) - 1
normal = np.stack(
    [nominal_sim.node_vars[key][:, background_time_idx] for key in norm_keys], axis=-1
)  # (n_nodes, 3), normal components xx, yy, zz
shear = np.stack(
    [nominal_sim.node_vars[key][:, background_time_idx] for key in dev_keys], axis=-1
)  # (n_nodes, 3), tensor shear components xy, xz, yz
strain_tensors = io.assemble_strain_tensor(
    normal,
    shear,
    shear_convention=io.EStrainShear.TENSOR,
)  # (n_nodes, 3, 3), symmetric small-strain tensor in global Cartesian axes
# Equivalent strain is sqrt(2/3 epsilon_dev:epsilon_dev): volumetric strain is
# excluded. Principal strains are descending; index zero is the first principal.
equivalent_strain = io.calc_equivalent_strain(strain_tensors)
first_principal = io.calc_principal_strains(strain_tensors)[:, 0]
background_fields = {
    "temperature": valvis.ValSimField(
        nominal_sim.node_vars["temperature"][:, background_time_idx],
        background_time_idx,
        sens.SensorDescriptor(name="Temperature", symbol="T", units="degC"),
    ),
    "strain_xx": valvis.ValSimField(
        normal[:, 0],
        background_time_idx,
        sens.SensorDescriptor(name="Strain xx", symbol=r"\varepsilon_{xx}", units="1"),
    ),
    "equivalent_strain": valvis.ValSimField(
        equivalent_strain,
        background_time_idx,
        sens.SensorDescriptor(
            name="Equivalent strain", symbol=r"\varepsilon_{eq}", units="1"
        ),
    ),
    "first_principal_strain": valvis.ValSimField(
        first_principal,
        background_time_idx,
        sens.SensorDescriptor(
            name="First principal strain", symbol=r"\varepsilon_1", units="1"
        ),
    ),
}
# Show all face tags through the translucent mesh, including the far faces.
point_labels = tuple(strain_sensors.get_descriptor().create_sensor_tags(len(coords)))
for background_name, background_field in background_fields.items():
    pv_plot = valvis.plot_val_metric_on_sim(
        val_result,
        strain_tag,
        val.EMetricType.MAVM,
        selectors={
            val.EValResAxis.COMPONENT: 0,
            val.EValResAxis.TIME: 4,
        },
        part=val.EValMetricPart.BALANCE,
        sim_data=nominal_sim,
        sim_field=background_field,
        sim_coord_units="mm",
        sim_coord_frame="cube",
        vis_opts=valvis.create_val_vis_opts(interactive=True) if SHOW_PLOTS else None,
        metric_opts=valvis.ValVisOpts(
            mesh_opacity=0.4, point_labels=point_labels, labels_always_visible=True
        ),
    )
    pv_plot.screenshot(output_path / f"strain_xx_balance_on_{background_name}.png")
    if SHOW_PLOTS:
        pv_plot.show()
    pv_plot.close()

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
