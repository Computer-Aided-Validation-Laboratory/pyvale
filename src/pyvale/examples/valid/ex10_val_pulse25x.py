# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Reproduce the Pulse25X point-sensor validation benchmark
================================================================================

This reproduces the point-sensor analysis in ``fullfieldvalmetrics`` for:

J. T. Horne-Jones et al. (2026), *Towards Virtual Qualification in Nuclear Fusion:
Demonstrating Probabilistic Model Validation on a High Heat Flux Component*,
`arXiv:2605.11886 <https://arxiv.org/abs/2605.11886>`_.

The full model contains 250 epistemic scenarios, each with 400 aleatory samples.
TC1--TC10 measure temperature; CV measures coil RMS voltage. The supplied CSVs
describe each channel as a scalar observation. GLOBAL geometry expresses this
channel-level statistical comparison.

This is a numerical reproduction of ``main_thermocouples_pulses25X.py`` as written.
That script retains only Pulse255 after overwriting earlier pulse data, assigns
uncertainty magnitudes by dictionary insertion order, and integrates unique
support values with equal weights. The example explicitly selects Pulse255,
associates offsets by insertion order and assigns equal mass to unique supports.
The packaged provenance records
source hashes, environment versions and the exact channel/offset association.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pyvale.data as dataset
import pyvale.sensorsim as sens
import pyvale.valid as val

matplotlib.use("Agg")

# %%
# Save all diagnostics by default; enable plotting and console inspection below.
SHOW_PLOTS = False
VERBOSE = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")
output_path = Path.cwd() / "pyvale-output" / "valid_ex10_pulse25x"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Load and transform data (pre-processing)
# -----------------------------------------
# The archive losslessly stores the full point-sensor CSV used by the paper.
# Axes: (11 channels, 250 transferable epistemic settings, 400 aleatory samples).
data_path = dataset.valid_data_dir()
with np.load(data_path / "pulse25x_full_simulation.npz") as saved:
    sensor_tags = tuple(saved["sensor_tags"].tolist())
    simulation = saved["simulation"]
with (data_path / "pulse25x_provenance.json").open() as stream:
    provenance = json.load(stream)

# %%
# The reference loads three pulses into the same dictionary: Pulse255 is the
# last assignment for every channel. Select it explicitly to reproduce that run.
# DIC-recorded thermocouples, HIVE TC2, and PICO voltage have different sample
# counts. Separate observables preserve each channel's own repeat count.
measurements: dict[str, np.ndarray] = {}
experimental_sources = (
    ("SteadyDICData", ("TC1", "TC3", "TC4", "TC5", "TC6", "TC7", "TC8", "TC9", "TC10")),
    ("SteadyHIVEData", ("TC2",)),
    ("SteadyPICOData", ("CV",)),
)
for source, channels in experimental_sources:
    frame = pd.read_csv(data_path / f"Pulse255_{source}.csv")
    for channel in channels:
        column = "Coil RMS Voltage" if channel == "CV" else channel
        measurements[channel] = frame[column].to_numpy(dtype=np.float64)

# %%
# Reproduce the original positional uncertainty association, including its
# mismatch with channel tags: TC3 receives TC2's magnitude, TC5 receives TC4's,
# etc. The exported channel_status.csv makes every association reviewable.
# These are coherent calibration shifts in degC or V, shared by every repeat.
summary = pd.read_csv(data_path / "pulse25x_uncertainty_summary.csv")
uncertainty_columns = (*sensor_tags[:-1], "Coil RMS Voltage")
uncertainties = summary.loc[summary.index[-1], list(uncertainty_columns)].to_numpy(
    dtype=float
)
geometry = val.ValGeometry(val.EValGeometry.GLOBAL, None)
observables: dict[str, val.ValObservable] = {}
sim_arrays: dict[str, val.ValArray] = {}
exp_arrays: dict[str, val.ValArray] = {}
channel_status = []
for index, (channel, observations) in enumerate(measurements.items()):
    units = "V" if channel == "CV" else "degC"
    observables[channel] = val.ValObservable(
        geometry,
        val.EValOperator.SAMPLE,
        (channel,),
        None,
        units,
    )
    sim_arrays[channel] = val.ValArray(
        simulation[sensor_tags.index(channel)],
        (val.EValAxis.EPISTEMIC_TRANS, val.EValAxis.ALEATORY),
    )
    missing = bool(np.any(np.isnan(observations)))
    # TC2 and CV have repeated readings. The legacy integration discards their
    # multiplicities. Explicit unique samples reproduce that weighting; ordinary
    # PyVale INCLUDE mode otherwise retains empirical duplicate probability mass.
    experimental_support = np.array([np.nan]) if missing else np.unique(observations)
    magnitude = uncertainties[index]
    exp_arrays[channel] = val.ValArray(
        experimental_support,
        (val.EValAxis.ALEATORY,),
        offsets=val.EpistemicOffsets(lower=magnitude, upper=magnitude),
    )
    channel_status.append(
        {
            "channel": channel,
            "units": units,
            "missing": missing,
            "raw_observations": len(observations),
            "unique_support": int(np.count_nonzero(np.isfinite(experimental_support))),
            "offset_source": sensor_tags[index],
            "offset_magnitude": magnitude,
        }
    )
val_case = val.ValCase(
    val.ValDomain(observables), val.ValData(sim_arrays), val.ValData(exp_arrays)
)
pd.DataFrame(channel_status).to_csv(output_path / "channel_status.csv", index=False)

# %%
# 2. Run validation analysis
# --------------------------
# Keep every epistemic setting and all three experimental offset positions.
# For each scalar channel the result shape is (point=1, component=1, time=1,
# sim_trans=250, exp_offset=3). Missing TC1, TC4 and TC7 retain NaN results.
# The reference uses t.ppf(0.95, n_unique-1), matching alpha=0.05 here. Its
# symmetric bounds use the 95th Student-t percentile for each support shift.
mavm = val.MetricMAVM(alpha=0.05, duplicates=val.EMAVMDuplicates.REJECT)
result = val.ValidationEngine([mavm]).run(val_case)
if VERBOSE:
    print("Simulation:", simulation.shape, "(channel, epis_trans, aleatory)")
    print(pd.DataFrame(channel_status))
    print("TC3 result axes:", result.get("TC3", val.EMetricType.MAVM).axes)

# %%
# 3. Post-processing and inspect data structures
# ----------------------------------------------
# The paper script picks WHOLE scenarios with the smallest/largest sum of ECDF
# support values. This is different from pointwise envelope bounds when CDFs
# cross. The nominal curve is the average of the conditional quantile functions.
# Both bound rules are available in PyVale; use SUPPORT_SUM for literal parity.
plot_opts = sens.PlotOptsGeneral(single_fig_scale=0.65)
metric_rows = [
    f"sim-{sim_name}_exp-{exp_name}-{direction}"
    for exp_name in ("min", "max")
    for sim_name in ("min", "max")
    for direction in ("d+", "d-")
]
metric_table: dict[str, list[float]] = {}
selected_scenarios: dict[str, dict[str, int]] = {}
axis_limits: dict[str, tuple[float, float]] = {}
for channel in measurements:
    units = observables[channel].units
    axis_label = "Coil RMS voltage [V]" if channel == "CV" else "Temperature [degC]"
    model_scenarios = val.extract_aleatory_scenarios(sim_arrays[channel])
    model_summary = val.calc_ecdf_summary(
        model_scenarios, bounds=val.EValECDFBounds.SUPPORT_SUM
    )
    selected_scenarios[channel] = {
        "lower": model_summary.lower_index,
        "upper": model_summary.upper_index,
    }

    # %%
    # 4. Analysis and visualisation
    # ------------------------------
    # Family plots retain the 250 conditional model ECDFs for every channel.
    # Available experimental readings retain their individual measurement spread.
    labels = val.ValPlotLabels(title=channel, x_label=axis_label)
    fig, ax = val.plot_val_ecdf_family(
        sim_arrays[channel],
        bounds=val.EValECDFBounds.SUPPORT_SUM,
        plot_opts=plot_opts,
        labels=labels,
    )
    figure_name = f"0_sim_allcdfs_{channel}.png"
    fig.savefig(output_path / figure_name, bbox_inches="tight")
    axis_limits[figure_name] = ax.get_xlim()
    if not SHOW_PLOTS:
        plt.close(fig)
    if np.any(np.isnan(exp_arrays[channel].data)):
        continue

    experimental_scenarios = val.extract_aleatory_scenarios(
        exp_arrays[channel], offset=None
    )
    experiment_summary = val.calc_ecdf_summary(
        experimental_scenarios,
        bounds=val.EValECDFBounds.SUPPORT_SUM,
    )
    model_lower = model_scenarios[model_summary.lower_index]
    model_upper = model_scenarios[model_summary.upper_index]
    exp_lower = exp_arrays[channel].select_aleatory(
        0, 0, 0, offset=val.EEpisOffset.LOWER
    )
    exp_upper = exp_arrays[channel].select_aleatory(
        0, 0, 0, offset=val.EEpisOffset.UPPER
    )
    correction = val.calc_mavm_correction(
        model_lower,
        model_upper,
        exp_lower,
        exp_upper,
        duplicates=val.EMAVMDuplicates.REJECT,
    )

    expected_correction = provenance["reference_selected_corrections"][channel]
    np.testing.assert_allclose(
        [correction.lower.d_minus, correction.upper.d_plus],
        [expected_correction["d_minus"], expected_correction["d_plus"]],
        rtol=1e-12,
        atol=1e-12,
    )

    # %%
    # Export the original four endpoint comparisons using the engine arrays.
    # These are distinct from the worst MAVM over every retained scenario.
    field = result.get(channel, val.EMetricType.MAVM)
    distances = []
    for offset in (val.EEpisOffset.LOWER, val.EEpisOffset.UPPER):
        for model_index in (model_summary.lower_index, model_summary.upper_index):
            distances.extend(
                [
                    float(field.d_plus[0, 0, 0, model_index, offset]),
                    float(field.d_minus[0, 0, 0, model_index, offset]),
                ]
            )
    metric_table[channel] = distances

    # %%
    # The next four figures reproduce the diagnostic families in the reference:
    # source distributions; endpoint correction cases; the outer correction with
    # experiment visible; and model-only uncertainty/correction for a clean view.
    # Gray fill shows the MAVM-corrected model support envelope.
    diagnostic_views = (
        ("1_exp-sim-cdfs", val.EValMAVMDiagnostic.DISTRIBUTIONS),
        ("2_dextremes_wcdfs", val.EValMAVMDiagnostic.COMPARISONS),
        ("3_dextremes_wexpcdfs", val.EValMAVMDiagnostic.CORRECTION),
        ("4_dextremes_simcdfsonly", val.EValMAVMDiagnostic.MODEL_CORRECTION),
    )
    for figure_prefix, view in diagnostic_views:
        fig, ax = val.plot_val_mavm_diagnostic(
            model_summary,
            experiment_summary,
            correction=correction,
            view=view,
            plot_opts=plot_opts,
            labels=labels,
        )
        figure_name = f"{figure_prefix}_{channel}.png"
        fig.savefig(output_path / figure_name, bbox_inches="tight")
        axis_limits[figure_name] = ax.get_xlim()
        if not SHOW_PLOTS:
            plt.close(fig)

# %%
# Check every d+ and d- against the independently executed legacy script's CSV:
# eight available channels × four endpoint pairs × two directions = 64 values.
# The frozen reference and its source hashes are bundled with the full dataset.
calculated = pd.DataFrame(metric_table, index=metric_rows)
reference = pd.read_csv(data_path / "pulse25x_reference_mavm.csv", index_col=0)
np.testing.assert_allclose(calculated, reference, rtol=1e-12, atol=1e-12)
calculated.to_csv(output_path / "pointsensors_mavm.csv")
parity = {
    "paper": provenance["paper"],
    "reference_script_sha256": provenance["reference_script_sha256"],
    "compared_values": int(calculated.size),
    "maximum_absolute_difference": float(
        np.max(np.abs(calculated.to_numpy() - reference.to_numpy()))
    ),
    "selected_scenarios": selected_scenarios,
}
with (output_path / "parity.json").open("w") as stream:
    json.dump(parity, stream, indent=2)
with (output_path / "axis_limits.json").open("w") as stream:
    json.dump(axis_limits, stream, indent=2)
if VERBOSE:
    print(calculated)
    print(parity)
if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
