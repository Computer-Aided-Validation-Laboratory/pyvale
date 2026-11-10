# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""MOOSE perturbations for 3D tensor validation
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
output_path = Path.cwd() / "pyvale-output" / "valid_ex8_cube_sweep"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Load and transform data (pre-processing)
# -----------------------------------------
# Run three small HEX8 cube models with the packaged Rabbit MOOSE application.
# One parameter changes at a time so its physical effect can be identified:
# heat flux controls the thermal load; thermal expansion couples temperature to
# mechanical strain. Both alternatives are +10% relative to the input deck.
# A fresh run directory preserves solver inputs/logs and makes reruns safe.
input_path = dataset.element_case_input_path(dataset.EElemTest.HEX8)
runner = mh.MooseRunner(timeout=120.0)
run_root = mh.get_next_run_path(output_path / "solves")
parameter_sets: dict[str, dict[str, mh.InputValues]] = {
    "nominal": {},
    "heat_flux_plus_10pct": {"surfHeatFlux": 1.1 * 4.67e6},
    "thermal_expansion_plus_10pct": {"ThermExp": 1.1 * 17.8e-6},
}
simulations = {}
for name, parameters in parameter_sets.items():
    solve_result = runner.run(
        input_path,
        input_params=parameters,
        output_dir=run_root / name,
    )
    simulations[name] = mh.ExodusLoader(solve_result.output_file).load_all_sim_data()
nominal_sim = simulations["nominal"]
experimental_sim = simulations["heat_flux_plus_10pct"]

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
for sim_data in simulations.values():
    sens.scale_length_units(1000.0, sim_data, disp_keys=("disp_x", "disp_y", "disp_z"))
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
# Keep all three solved parameter settings as separate transferable epistemic
# alternatives. They must not be pooled into an aleatory sample distribution.
model = val.val_from_sim(
    {"strain": strain_field},
    domain,
    [((index,), sim_data) for index, sim_data in enumerate(simulations.values())],
    sample_axes=(val.EValAxis.EPISTEMIC_TRANS,),
    sample_shape=(3,),
)
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
        selectors={val.EValResAxis.COMPONENT: component, val.EValResAxis.SIM_TRANS: 0},
        direction="d_minus",
        plot_opts=plot_opts,
        labels=val.ValPlotLabels(title=component_key),
    )
    fig.savefig(output_path / f"{component_key}_d_minus_time.png", bbox_inches="tight")
    if not SHOW_PLOTS:
        plt.close("all")

# %%
# Compare the parameter alternatives without collapsing their result axis.
# The filenames identify the physical perturbation; y labels identify total AVM.
for scenario, scenario_name in enumerate(parameter_sets):
    fig, axes = val.plot_val_metric_time_points(
        result,
        "strain",
        val.EMetricType.AVM,
        points=points,
        selectors={val.EValResAxis.COMPONENT: 0, val.EValResAxis.SIM_TRANS: scenario},
        plot_opts=plot_opts,
        labels=val.ValPlotLabels(title=f"strain_xx: {scenario_name}"),
    )
    fig.savefig(output_path / f"{scenario_name}_avm_time.png", bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
