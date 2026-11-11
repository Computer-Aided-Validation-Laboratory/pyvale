# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Validation of 3D cube tensor observations
================================================================================

Follow the six cube-face sensor locations used in extsensorsim/ex3f_tens3d.
Synthetic measurements illustrate the API; they are not independent evidence
that this model represents a real experiment.
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

matplotlib.use("Agg")

# %%
# Save figures without windows or console output unless explicitly requested.
SHOW_PLOTS = False
VERBOSE = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")
output_path = Path.cwd() / "pyvale-output" / "valid_ex7_cube_tensor"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Load and transform data (pre-processing)
# -----------------------------------------
# The packaged HEX20 data describe a 10 mm cube. Exodus stores coordinates
# in metres and tensor strains as dimensionless components. Scale lengths only;
# strain must retain its physical values.
input_path = dataset.element_case_output_path(dataset.EElemTest.HEX20)
nominal_sim = mh.ExodusLoader(input_path).load_all_sim_data()
experimental_sim = nominal_sim

# %%
# Six strain rosettes sit at the centres of the xz, xy and yz face pairs.
# All tensors use global Cartesian axes here (no sensor rotations). Thus model
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
domain = val.ValDomain(
    {
        "strain": val.ValObservable(
            val.ValGeometry(
                val.EValGeometry.POINT, coords, coordinate_units="mm", frame="cube"
            ),
            val.EValOperator.SAMPLE,
            component_keys,
            times,
            "1",
            time_units="s",
        )
    }
)
strain_field = sens.FieldTensor(nominal_sim, norm_keys, dev_keys, sens.EDim.THREED)
model = val.val_from_sim({"strain": strain_field}, domain, [((), nominal_sim)])
experiment = val.ValData(
    {
        "strain": val.ValArray(
            measurements,
            (
                val.EValAxis.POINT,
                val.EValAxis.COMPONENT,
                val.EValAxis.TIME,
                val.EValAxis.ALEATORY,
            ),
        )
    }
)
val_case = val.ValCase(domain, model, experiment)

# %%
# 2. Run validation analysis
# --------------------------
# The model is deterministic within each selected parameter scenario, while
# the experiment has six repeat observations. INCLUDE handles repeated supports.
mavm = val.MetricMAVM(duplicates=val.EMAVMDuplicates.INCLUDE)
result = val.ValidationEngine([val.MetricAVM(), mavm]).run(val_case)

# %%
# 3. Post-processing and inspect data structures
# ----------------------------------------------
# No tensor component is flattened into the observation axis. Result axes start
# (6 points, 6 components, 5 times), followed by retained parameter scenarios.
field = result.get("strain", val.EMetricType.MAVM)
if VERBOSE:
    print(
        "Model axes/shape:",
        model.quantities["strain"].axes,
        model.quantities["strain"].data.shape,
    )
    print("Experiment shape:", measurements.shape)
    print("Result axes/shape:", field.axes, field.metric.shape)

# %%
# 4. Analysis and visualisation
# ------------------------------
# Show all six face sites for every tensor component. The nominal model is
# selected explicitly in a parameter sweep, so these panels compare like cells.
plot_opts = sens.PlotOptsGeneral()
points = tuple(range(len(coords)))
for component, component_key in enumerate(component_keys):
    fig, axes = val.plot_val_cdfs_points(
        val_case,
        result,
        "strain",
        points=points,
        component=component,
        time=4,
        metric=mavm,
        plot_opts=plot_opts,
        labels=val.ValPlotLabels(x_label=f"{component_key} [1]"),
    )
    fig.savefig(output_path / f"{component_key}_cdfs.png", bbox_inches="tight")
    fig, axes = val.plot_val_metric_time_points(
        result,
        "strain",
        val.EMetricType.MAVM,
        points=points,
        selectors={val.EValResAxis.COMPONENT: component},
        part=val.EValMetricPart.D_MINUS,
        plot_opts=plot_opts,
        labels=val.ValPlotLabels(title=component_key),
    )
    fig.savefig(output_path / f"{component_key}_d_minus_time.png", bbox_inches="tight")
    if not SHOW_PLOTS:
        plt.close("all")
# %%
# Overlay the selected strain_xx metric on several independent background fields.
# The two colourbars deliberately identify different quantities: model field
# values on the mesh, and validation discrepancy at the six original face sites.
# Every field is evaluated at the final simulation step, matching validation t.
background_step = len(nominal_sim.time) - 1
normal = np.stack(
    [nominal_sim.node_vars[key][:, background_step] for key in norm_keys], axis=-1
)  # (n_nodes, 3), normal components xx, yy, zz
shear = np.stack(
    [nominal_sim.node_vars[key][:, background_step] for key in dev_keys], axis=-1
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
    "temperature": val.ValSimField(
        nominal_sim.node_vars["temperature"][:, background_step],
        background_step,
        sens.SensorDescriptor(name="Temperature", symbol="T", units="degC"),
    ),
    "strain_xx": val.ValSimField(
        normal[:, 0],
        background_step,
        sens.SensorDescriptor(name="Strain xx", symbol=r"\varepsilon_{xx}", units="1"),
    ),
    "equivalent_strain": val.ValSimField(
        equivalent_strain,
        background_step,
        sens.SensorDescriptor(
            name="Equivalent strain", symbol=r"\varepsilon_{eq}", units="1"
        ),
    ),
    "first_principal_strain": val.ValSimField(
        first_principal,
        background_step,
        sens.SensorDescriptor(
            name="First principal strain", symbol=r"\varepsilon_1", units="1"
        ),
    ),
}
scene_opts = sens.VisOptsSimSensors(
    interactive=SHOW_PLOTS,
    camera_position="iso",
    sens_point_size=24,
    sens_label_font_size=16,
    colour_bar_vertical=False,
    colour_bar_font_size=13,
)
# Show all face tags through the translucent mesh, including the far faces.
point_labels = tuple(strain_sensors.get_descriptor().create_sensor_tags(len(coords)))
for background_name, background_field in background_fields.items():
    pv_plot = val.plot_val_metric_on_sim(
        result,
        "strain",
        val.EMetricType.MAVM,
        selectors={
            val.EValResAxis.COMPONENT: 0,
            val.EValResAxis.TIME: 4,
        },
        part=val.EValMetricPart.BALANCE,
        sim_data=nominal_sim,
        sim_field=background_field,
        sim_coordinate_units="mm",
        sim_frame="cube",
        vis_opts=scene_opts,
        metric_opts=val.ValVisOpts(
            mesh_opacity=0.4, point_labels=point_labels, labels_always_visible=True
        ),
    )
    pv_plot.screenshot(output_path / f"strain_xx_balance_on_{background_name}.png")
    if SHOW_PLOTS:
        pv_plot.show()
    pv_plot.close()

# %%
# Additional distribution diagnostics
# ------------------------------------
# Inspect every sensor at the same physical time. Whole model scenarios are
# selected by support sum for the endpoint correction; this is distinct from
# the pointwise envelope and from the largest MAVM over all scenarios.
# Calibration offsets, when present, shift an entire conditional sample coherently.
model_quantity = val_case.sim.quantities["strain"]
experiment_quantity = val_case.exp.quantities["strain"]
panel_opts = sens.PlotOptsGeneral(font_leg_size=6)
diagnostic_views = (val.EValMAVMDiagnostic.CORRECTION,)
for view in diagnostic_views:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    for point, ax in enumerate(axes.flat):
        model_scenarios = val.extract_aleatory_scenarios(
            model_quantity,
            point=point,
            component=0,
            time=4,
        )
        experimental_scenarios = val.extract_aleatory_scenarios(
            experiment_quantity,
            point=point,
            component=0,
            time=4,
            offset=None,
        )
        model_summary = val.calc_ecdf_summary(
            model_scenarios,
            bounds=val.EValECDFBounds.SUPPORT_SUM,
        )
        experiment_summary = val.calc_ecdf_summary(
            experimental_scenarios,
            bounds=val.EValECDFBounds.SUPPORT_SUM,
        )
        correction = val.calc_mavm_correction(
            model_scenarios[model_summary.lower_index],
            model_scenarios[model_summary.upper_index],
            experimental_scenarios[experiment_summary.lower_index],
            experimental_scenarios[experiment_summary.upper_index],
        )
        val.plot_val_mavm_diagnostic(
            model_summary,
            experiment_summary,
            correction=correction,
            view=view,
            plot_opts=panel_opts,
            ax=ax,
            labels=val.ValPlotLabels(
                title=f"Point {point}: MAVM support shifts", x_label="Strain xx [1]"
            ),
        )
    fig.savefig(output_path / f"mavm_{view.name.lower()}.png", bbox_inches="tight")
    if not SHOW_PLOTS:
        plt.close(fig)

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
