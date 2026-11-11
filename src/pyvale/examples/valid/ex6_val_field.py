# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Convert prepared SensorSim runs and sample real SimData on a 2D grid.

This example compares nominal plate fields with synthetic measurements from
the packaged plate whose conductivity and thermal expansion are both 10% lower.
It covers the direct SensorSim-to-ExpData bridge and IField sampling without
running MOOSE or regenerating the SensorSim experiment.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyvale.data as dataset
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.valid as val

matplotlib.use("Agg")

# %%
# Save figures silently by default. Enable local interactive windows explicitly.
SHOW_PLOTS = False
VERBOSE = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")
output_path = Path.cwd() / "pyvale-output" / "valid_ex6_field"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Load and transform data (pre-processing)
# -----------------------------------------
# Twelve thermocouples and in-plane strain rosettes lie on a 4 by 3 grid on the
# plate. Coordinates are (12 points, xyz) in mm; times are (5,) in seconds.
# Saved measurements have axes (6 repeats, 12 points, 1 or 3 components, 5 times).
# The experimental simulation has conductivity and thermal expansion 10% lower.
# Retain each run's sensor/component/time identity while converting to ExpData.

with np.load(dataset.valid_data_dir() / "plate_validation_2d.npz") as saved:
    coords = saved["coords"].copy()
    times = saved["times"].copy()
    output = {
        ("perturbed", "temperature", "meas"): saved["perturbed_temperature"].copy(),
        ("perturbed", "strain", "meas"): saved["perturbed_strain"].copy(),
        ("perturbed", "temperature", "sens_times"): times,
        ("perturbed", "strain", "sens_times"): times,
    }

# %%
# A selection maps an observable to a sensor-array key, component index and
# coordinates. ExpData retains acquisition identity; combine runs only after
# registering the common comparison domain.
selections = {
    "temperature": ("temperature", 0, coords),
    "strain_xx": ("strain", 0, coords),
    "strain_yy": ("strain", 1, coords),
    "strain_xy": ("strain", 2, coords),
}
acquisitions = val.conv_exp_sim_to_data(
    output,
    sim_key="perturbed",
    selections=selections,
)

# %%
# Register the same locations and times for the model and every acquisition.
# Splitting rosette components into separate observables makes their units and
# component mapping explicit at the conversion boundary.
geometry = val.ValGeometry(
    val.EValGeometry.POINT, coords, coordinate_units="mm", frame="plate"
)
domain = val.ValDomain(
    {
        name: val.ValObservable(
            geometry,
            val.EValOperator.SAMPLE,
            (name,),
            times,
            "degC" if name == "temperature" else "1",
            time_units="s",
        )
        for name in selections
    }
)

# %%
# Sample actual Exodus fields at those sensor sites rather than comparing mesh
# nodes directly. Convert metres to mm to match the acquisition coordinates.
sim_path = dataset.thermomechanical_2d_experiment_paths()[0]
sim_data = mh.ExodusLoader(sim_path).load_all_sim_data()
sens.scale_length_units(1000.0, sim_data, disp_keys=("disp_x", "disp_y"))
temperature = sens.FieldScalar(sim_data, "temperature", sens.EDim.TWOD)
strain = sens.FieldTensor(
    sim_data,
    ("strain_xx", "strain_yy"),
    ("strain_xy",),
    sens.EDim.TWOD,
)
fields = {
    "temperature": temperature,
    "strain_xx": strain,
    "strain_yy": strain,
    "strain_xy": strain,
}
simulation = val.val_from_sim(fields, domain, [((), sim_data)])
cases = [
    val.ValCase(domain, simulation, val.val_from_exp(run, domain))
    for run in acquisitions
]
val_case = val.combine_val_cases(*cases)
# %%
# 2. Run validation analysis
# --------------------------
# The bridge appends the repeated acquisitions on the aleatory axis; physical
# axes remain (point, component, time), followed by uncertainty axes.
result = val.ValidationEngine([val.MetricAVM(), val.MetricMAVM()]).run(val_case)

# %%
# 3. Post-processing and inspect data structures
# ----------------------------------------------
# Field samples and converted acquisitions share (12, 1, 5) physical shape.
# The experiment additionally has six aleatory observations per physical cell.
model_quantity = val_case.sim.quantities["temperature"]
experiment_quantity = val_case.exp.quantities["temperature"]
if VERBOSE:
    print("Model:", model_quantity.axes, model_quantity.data.shape)
    print("Experiment:", experiment_quantity.axes, experiment_quantity.data.shape)

# %%
# 4. Analysis and visualisation
# ------------------------------
# AVM strain has dimensionless strain units. Show all thermocouple histories
# and CDFs so spatial differences are visible without editing a point index.
plot_opts = sens.PlotOptsGeneral()
# %%
# Overlay validation points on the nominal model in its original xyz geometry.
# The model mesh and sensor positions must share mm coordinates and the plate
# frame. Background nodal scalars (n_nodes,) use their own physical colourbar.
background_step = len(sim_data.time) - 1
background_field = val.ValSimField(
    sim_data.node_vars["strain_xx"][:, background_step],
    background_step,
    sens.SensorDescriptor(
        name="Axial strain xx", symbol=r"\varepsilon_{xx}", units="1"
    ),
)
scene_opts = sens.VisOptsSimSensors(
    interactive=SHOW_PLOTS,
    camera_position="xy",
    sens_point_size=18,
    sens_label_font_size=14,
    colour_bar_vertical=False,
    colour_bar_font_size=13,
)
point_labels = tuple(sens.SensorDescriptor(tag="SG").create_sensor_tags(len(coords)))
# Total AVM is a nonnegative magnitude, so its point scale starts at zero.
pv_plot = val.plot_val_metric_on_sim(
    result,
    "strain_xx",
    val.EMetricType.AVM,
    selectors={
        val.EValResAxis.TIME: 4,
    },
    part=val.EValMetricPart.TOTAL,
    sim_data=sim_data,
    sim_field=background_field,
    sim_coordinate_units="mm",
    sim_frame="plate",
    vis_opts=scene_opts,
    metric_opts=val.ValVisOpts(point_labels=point_labels, labels_always_visible=True),
)
pv_plot.screenshot(output_path / "strain_xx_avm_on_sim.png")
if SHOW_PLOTS:
    pv_plot.show()
pv_plot.close()


points = tuple(range(len(coords)))
fig, axes = val.plot_val_metric_time_points(
    result,
    "temperature",
    val.EMetricType.AVM,
    points=points,
    plot_opts=plot_opts,
)
fig.savefig(output_path / "temperature_avm_time.png", bbox_inches="tight")
fig, axes = val.plot_val_cdfs_points(
    val_case,
    result,
    "temperature",
    points=points,
    time=4,
    metric=val.MetricMAVM(),
    plot_opts=plot_opts,
)
fig.savefig(output_path / "temperature_cdfs.png", bbox_inches="tight")
# %%
# Additional distribution diagnostics
# ------------------------------------
# Inspect every sensor at the same physical time. Whole model scenarios are
# selected by support sum for the endpoint correction; this is distinct from
# the pointwise envelope and from the largest MAVM over all scenarios.
# Calibration offsets, when present, shift an entire conditional sample coherently.
model_quantity = val_case.sim.quantities["strain_xx"]
experiment_quantity = val_case.exp.quantities["strain_xx"]
panel_opts = sens.PlotOptsGeneral(font_leg_size=6)
diagnostic_views = (val.EValMAVMDiagnostic.CORRECTION,)
for view in diagnostic_views:
    fig, axes = plt.subplots(4, 3, figsize=(12, 16), layout="constrained")
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
