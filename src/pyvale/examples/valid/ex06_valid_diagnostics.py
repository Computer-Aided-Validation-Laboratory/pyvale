# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Interpreting validation diagnostics
================================================================================

Interpret conditional CDF families, uncertainty envelopes, directional metric
distributions and MAVM support-shift diagnostics. This standalone example
combines sampled model scenarios, time-dependent data and calibration offsets.
"""

from pathlib import Path
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.figure import Figure

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
from pyvale import valid
from pyvale.valid import validplot, validvis

matplotlib.use("Agg")

# %%
# Console inspection is printed below. Enable interactive windows with SHOW_PLOTS.
SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")
output_path = Path.cwd() / "pyvale-output" / "valid_ex6_diagnostics"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Load and transform data
# -----------------------------------------
# The plate has 12 sensor locations on a 4 by 3 grid: x=10..90 mm,
# y=10..40 mm, z=0. Each site has a thermocouple (one temperature component)
# and an in-plane strain rosette (xx, yy, xy). Six virtual repeat experiments
# supply aleatory samples at five common physical times.
# The second simulation has conductivity and thermal expansion reduced by 10%.
# We treat its noisy measurements as the experiment to expose model discrepancy.
# Prepared arrays already follow the validation axis order declared below.
# Temperature has one component; strain has three components (xx, yy, xy).
with np.load(dataset.valid_data_dir() / "plate_validation_2d.npz") as saved:
    coords: np.ndarray = saved["coords"].copy()
    times: np.ndarray = saved["times"].copy()
    sim_temp: np.ndarray = saved["sim_temp"].copy()  # (12, 1, 5, 2, 6)
    exp_temp: np.ndarray = saved["exp_temp"].copy()  # (12, 1, 5, 6)
    sim_strain: np.ndarray = saved["sim_strain"].copy()
    exp_strain: np.ndarray = saved["exp_strain"].copy()

# %%
# A shared observables registers units, component order and physical coordinates.
# Validation compares corresponding physical cells, retaining distinct point,
# component, time and repeat-experiment axes.
temperature_tag = "plate_temperature"
strain_tag = "plate_strain"
observables = {
    temperature_tag: valid.ValidObservable(
        components=("temperature",),
        geometry=valid.EValidGeometry.POINT,
        coords=coords,
        times=times,
        phys_units="degC",
        coord_units="mm",
        time_units="s",
    ),
    strain_tag: valid.ValidObservable(
        components=("strain_xx", "strain_yy", "strain_xy"),
        geometry=valid.EValidGeometry.POINT,
        coords=coords,
        times=times,
        phys_units="1",
        coord_units="mm",
        time_units="s",
    ),
}

# EPISTEMIC_TRANS holds two alternative model parameter settings: nominal
# then perturbed. Each setting contains its own six aleatory observations.
# Physical axes identify each comparison cell; ALEATORY holds its repeated
# observations. The declared axis order matches the arrays in the archive.
sim_axes = (
    valid.EValidAxis.POINT,
    valid.EValidAxis.COMPONENT,
    valid.EValidAxis.TIME,
    valid.EValidAxis.EPISTEMIC_TRANS,
    valid.EValidAxis.ALEATORY,
)
exp_axes = (
    valid.EValidAxis.POINT,
    valid.EValidAxis.COMPONENT,
    valid.EValidAxis.TIME,
    valid.EValidAxis.ALEATORY,
)
sim_arrays: dict[str, valid.ValidArray] = {}
exp_arrays: dict[str, valid.ValidArray] = {}
for observable_tag, sim_values, exp_values in (
    (temperature_tag, sim_temp, exp_temp),
    (strain_tag, sim_strain, exp_strain),
):
    # Lower and upper magnitudes describe a coherent calibration shift in
    # observable units, shared by every repeat at a physical comparison cell.
    magnitude = 0.5 if observable_tag == temperature_tag else 4.0e-6
    if observable_tag == temperature_tag:
        spatial = coords[:, 0, None] / 100.0  # (point, singleton time)
        temporal = times[None, :] / times[-1]  # (singleton point, time)
        bound = valid.EpistemicOffsets(
            lower=magnitude * (1.0 + 0.2 * spatial + 0.2 * temporal),
            upper=1.5 * magnitude * (1.0 + 0.1 * spatial + 0.3 * temporal),
            axes=(valid.EValidAxis.POINT, valid.EValidAxis.TIME),
        )
    else:
        bound = valid.EpistemicOffsets(lower=magnitude, upper=1.5 * magnitude)
    sim_arrays[observable_tag] = valid.ValidArray(sim_values, sim_axes)
    exp_arrays[observable_tag] = valid.ValidArray(exp_values, exp_axes, offsets=bound)

valid_case = valid.ValidCase(observables, sim_arrays, exp_arrays)

# %%
# 2. Run validation analysis
# --------------------------
# AVM gives a distance in the observable's units. MAVM additionally separates
# positive and negative support mismatch; INCLUDE retains repeated observations.
valid_result: valid.ValidResult = valid.ValidEngine(
    [valid.MetricAVM(), valid.MetricMAVM()]
).run(valid_case)

# %%
# 3. Post-processing and inspect data structures
# ----------------------------------------------
# metric_field.axes names each array dimension. The first three axes are always
# (point, component, time), including singleton time for steady comparisons.
# Non-singleton epistemic/offset axes follow in the engine's canonical order.
metric_field: valid.ValidMetricField = valid_result.metrics[temperature_tag][
    valid.EMetricType.MAVM
]
metric_shape = metric_field.outputs[valid.EMetricOutput.D_PLUS].shape
metric_axes = metric_field.axes
print("Simulation axes:", sim_arrays[temperature_tag].axes)
print("Simulation shape:", sim_arrays[temperature_tag].data.shape)
print("Result axes:", metric_axes, "shape:", metric_shape)

# %%
# 4. Analysis and visualisation
# ------------------------------
# Use the SensorSim appearance options so fonts, resolution and line styles
# agree with sensor figures. Each CDF panel overlays simulation and experiment;
# dashed curves show the experimental MAVM confidence bounds.
point_idxs = tuple(range(len(coords)))
fig: Figure
axes: np.ndarray
fig, axes = validplot.plot_valid_cdfs_points(
    valid_case,
    valid_result,
    temperature_tag,
    point_idxs=point_idxs,
    time_idx=4,
    metric=valid.MetricMAVM(),
    valid_plot_opts=validplot.ValidPlotOpts(x_label="Temperature [degC]"),
)
fig.savefig(output_path / "temperature_cdfs.png", bbox_inches="tight")

# %%
# Envelopes describe all retained scenarios at a site. They are pointwise ECDF
# bounds over the conditional distributions. Show every plate sensor.
fig, axes = validplot.plot_valid_cdf_envelope_points(
    valid_case,
    temperature_tag,
    source=valid.EValidSource.EXP,
    time_idx=4,
    valid_plot_opts=validplot.ValidPlotOpts(x_label="Temperature [degC]"),
)
fig.savefig(output_path / "temperature_envelopes.png", bbox_inches="tight")

# %%
# Each panel follows one physical sensor through time. Explicit selectors fix
# any remaining epistemic scenario; d+ identifies the directional MAVM quantity.
fig, axes = validplot.plot_valid_metric_time_points(
    valid_result,
    temperature_tag,
    valid.EMetricType.MAVM,
    point_idxs=point_idxs,
    selectors={
        valid.EValidResAxis.SIM_TRANS: 0,
        valid.EValidResAxis.EXP_OFFSET: valid.EEpisOffset.UPPER,
    },
    output=valid.EMetricOutput.D_PLUS,
)
fig.savefig(output_path / "temperature_d_plus_time.png", bbox_inches="tight")

# %%
# Overlay validation points on the nominal model in its original xyz geometry.
# The model mesh and sensor positions use the same coordinates in mm.
# Background nodal scalars (n_nodes,) have a physical colourbar.
sim_path = dataset.thermomechanical_2d_experiment_paths()[0]
sim_data: io.SimData = mh.ExodusLoader(sim_path).load_all_sim_data()
sens.scale_length_units(1000.0, sim_data, disp_keys=("disp_x", "disp_y"))
assert sim_data.time is not None
background_time_idx = sim_data.time.size - 1
# Show the MAVM d- output directly, with a nonnegative colour scale.
# The d+ output is stored separately and can be selected in the same way.
pv_plot: pv.Plotter = validvis.plot_valid_metric_on_sim(
    valid_result,
    temperature_tag,
    valid.EMetricType.MAVM,
    selectors={
        valid.EValidResAxis.SIM_TRANS: 0,
        valid.EValidResAxis.TIME: 4,
        valid.EValidResAxis.EXP_OFFSET: valid.EEpisOffset.UPPER,
    },
    output=valid.EMetricOutput.D_MINUS,
    sim_data=sim_data,
    sim_field="temperature",
    sim_time_idx=background_time_idx,
    descriptor=sens.SensorDescriptor(
        name="Temperature", symbol=r"T", units="degC", tag="TC"
    ),
    sim_coord_units="mm",
    vis_opts=validvis.create_valid_scene_opts(interactive=SHOW_PLOTS),
    metric_opts=validvis.ValidMetricVisOpts(labels_always_visible=True),
)
pv_plot.screenshot(output_path / "temperature_mavm_d_minus_on_sim.png")
if SHOW_PLOTS:
    pv_plot.show()
pv_plot.close()
# %%
# Figures are saved even when interactive plotting is disabled. Keeping the
# complete analysis in its own directory prevents clashes with other examples.
# %%
# Additional distribution diagnostics
# ------------------------------------
# Inspect every sensor at the same physical time. Whole model scenarios are
# selected by support sum for the endpoint correction; this is distinct from
# the pointwise envelope and from the largest MAVM over all scenarios.
# Calibration offsets, when present, shift an entire conditional sample coherently.
# Start with the conditional samples. extract_aleatory_scenarios accepts
# epis_trans_idx and epis_nuis_idx to inspect a chosen parameter setting; None retains
# every setting. offset=None includes nominal, lower and upper calibration
# shifts. Every returned array is one conditional aleatory sample.
model_quantity = valid_case.sim[temperature_tag]
exp_quantity = valid_case.exp[temperature_tag]
# Pointwise envelopes choose extrema at each probability and describe the full
# family spread. SUPPORT_SUM selects whole scenarios by the sum of their unique
# support values; lower_idx and upper_idx identify those original samples.
# Quantile averaging supplies a representative curve through the family.
fig, axes = validplot.create_valid_plot_grid(len(coords))
for point_idx, ax in enumerate(axes.flat):
    validplot.plot_valid_ecdf_family(
        model_quantity,
        point_idx=point_idx,
        time_idx=4,
        ax=ax,
        valid_plot_opts=validplot.ValidPlotOpts(
            title=f"Point {point_idx}: conditional model CDFs",
            x_label="Temperature [degC]",
        ),
    )
fig.savefig(output_path / "conditional_cdf_families.png", bbox_inches="tight")
if not SHOW_PLOTS:
    plt.close(fig)

# Read the four views in order: DISTRIBUTIONS compares model and experiment;
# COMPARISONS shows all four lower/upper endpoint pairings; CORRECTION overlays
# the selected directional support shifts; MODEL_CORRECTION shows the resulting
# model support envelope. The shifts retain the observable's physical units.
# d+ extends the upper model support and d- extends the lower model support.
diagnostic_views = (
    valid.EValidMAVMDiagnostic.DISTRIBUTIONS,
    valid.EValidMAVMDiagnostic.COMPARISONS,
    valid.EValidMAVMDiagnostic.CORRECTION,
    valid.EValidMAVMDiagnostic.MODEL_CORRECTION,
)
for view in diagnostic_views:
    fig, axes = validplot.create_valid_plot_grid(len(coords))
    for point_idx, ax in enumerate(axes.flat):
        model_scenarios: list[np.ndarray] = valid.extract_aleatory_scenarios(
            model_quantity,
            point_idx=point_idx,
            time_idx=4,
        )
        exp_scenarios: list[np.ndarray] = valid.extract_aleatory_scenarios(
            exp_quantity,
            point_idx=point_idx,
            time_idx=4,
            offset=None,
        )
        model_summary: valid.ValidECDFSummary = valid.calc_ecdf_summary(
            model_scenarios,
            bounds=valid.EValidECDFBounds.SUPPORT_SUM,
        )
        exp_summary: valid.ValidECDFSummary = valid.calc_ecdf_summary(
            exp_scenarios,
            bounds=valid.EValidECDFBounds.SUPPORT_SUM,
        )
        correction: valid.MAVMCorrection = valid.calc_mavm_correction(
            model_scenarios[cast(int, model_summary.lower_idx)],
            model_scenarios[cast(int, model_summary.upper_idx)],
            exp_scenarios[cast(int, exp_summary.lower_idx)],
            exp_scenarios[cast(int, exp_summary.upper_idx)],
        )
        validplot.plot_valid_mavm_diagnostic(
            model_summary,
            exp_summary,
            correction=correction,
            view=view,
            ax=ax,
            valid_plot_opts=validplot.ValidPlotOpts(
                title=f"Point {point_idx}: MAVM support shifts",
                x_label="Temperature [degC]",
            ),
        )
    fig.savefig(output_path / f"mavm_{view.name.lower()}.png", bbox_inches="tight")
    if not SHOW_PLOTS:
        plt.close(fig)

# %%
# Histograms count physical/scenario cells within each distance bin. Separate
# directional magnitudes expose asymmetry between d+ and d-.
fig, axes = validplot.create_valid_plot_grid(2)
for output, ax in zip(
    (valid.EMetricOutput.D_PLUS, valid.EMetricOutput.D_MINUS), axes.flat, strict=True
):
    validplot.plot_valid_metric_distribution(
        valid_result,
        temperature_tag,
        valid.EMetricType.MAVM,
        output=output,
        ax=ax,
        valid_plot_opts=validplot.ValidPlotOpts(
            title="Temperature: retained physical/scenario cells"
        ),
    )
fig.savefig(output_path / "directional_metric_distributions.png", bbox_inches="tight")
if not SHOW_PLOTS:
    plt.close(fig)

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
