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

from pathlib import Path
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import pyvale.data as dataset
import pyvale.validation as vl

matplotlib.use("Agg")

# %%
# Save diagnostics and print data structures; enable interactive windows below.
SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")
output_path = Path.cwd() / "pyvale-output" / "valid_ex10_pulse25x"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# 1. Load and transform data
# -----------------------------------------
# The archive losslessly stores the full point-sensor CSV used by the paper.
# Axes: (11 channels, 250 transferable epistemic settings, 400 aleatory samples).
data_path: Path = dataset.valid_data_dir()
with np.load(data_path / "pulse25x_full_simulation.npz") as saved:
    sensor_tags = tuple(saved["sensor_tags"].tolist())
    sim_data: np.ndarray = saved["simulation"]
# %%
# The reference loads three pulses into the same dictionary: Pulse255 is the
# last assignment for every channel. Select it explicitly to reproduce that run.
# DIC-recorded thermocouples, HIVE TC2, and PICO voltage have different sample
# counts. Separate observables preserve each channel's own repeat count.
measurements: dict[str, np.ndarray] = {}
exp_sources = (
    ("SteadyDICData", ("TC1", "TC3", "TC4", "TC5", "TC6", "TC7", "TC8", "TC9", "TC10")),
    ("SteadyHIVEData", ("TC2",)),
    ("SteadyPICOData", ("CV",)),
)
for source, channels in exp_sources:
    measurement_table: pd.DataFrame = pd.read_csv(data_path / f"Pulse255_{source}.csv")
    for channel in channels:
        column = "Coil RMS Voltage" if channel == "CV" else channel
        measurements[channel] = measurement_table[column].to_numpy(dtype=np.float64)

# %%
# Reproduce the original positional uncertainty association, including its
# mismatch with channel tags: TC3 receives TC2's magnitude, TC5 receives TC4's,
# etc. The exported channel_status.csv makes every association reviewable.
# These are coherent calibration shifts in degC or V, shared by every repeat.
summary: pd.DataFrame = pd.read_csv(data_path / "pulse25x_uncertainty_summary.csv")
uncertainty_columns = (*sensor_tags[:-1], "Coil RMS Voltage")
uncertainties = summary.loc[summary.index[-1], list(uncertainty_columns)].to_numpy(
    dtype=float
)
observables: dict[str, vl.ValidObservable] = {}
sim_arrays: dict[str, vl.ValidArray] = {}
exp_arrays: dict[str, vl.ValidArray] = {}
channel_status = []
for index, (channel, observations) in enumerate(measurements.items()):
    units = "V" if channel == "CV" else "degC"
    observables[channel] = vl.ValidObservable(
        components=(channel,),
        geometry=vl.EValidGeometry.GLOBAL,
        phys_units=units,
    )
    sim_arrays[channel] = vl.ValidArray(
        sim_data[sensor_tags.index(channel)],
        (vl.EValidAxis.EPISTEMIC_TRANS, vl.EValidAxis.ALEATORY),
    )
    missing = bool(np.any(np.isnan(observations)))
    # TC2 and CV have repeated readings. The legacy integration discards their
    # multiplicities. Explicit unique samples reproduce that weighting; ordinary
    # PyVale INCLUDE mode otherwise retains empirical duplicate probability mass.
    exp_support = np.array([np.nan]) if missing else np.unique(observations)
    magnitude = uncertainties[index]
    exp_arrays[channel] = vl.ValidArray(
        exp_support,
        (vl.EValidAxis.ALEATORY,),
        offsets=vl.EpistemicOffsets(lower=magnitude, upper=magnitude),
    )
    channel_status.append(
        {
            "channel": channel,
            "units": units,
            "missing": missing,
            "raw_observations": len(observations),
            "unique_support": int(np.count_nonzero(np.isfinite(exp_support))),
            "offset_source": sensor_tags[index],
            "offset_magnitude": magnitude,
        }
    )
valid_case = vl.ValidCase(observables, sim_arrays, exp_arrays)
pd.DataFrame(channel_status).to_csv(output_path / "channel_status.csv", index=False)

# %%
# 2. Run validation analysis
# --------------------------
# Keep every epistemic setting and all three experimental offset positions.
# For each scalar channel the result shape is (point=1, component=1, time=1,
# sim_trans=250, exp_offset=3). Missing TC1, TC4 and TC7 retain NaN results.
# The reference uses t.ppf(0.95, n_unique-1), matching alpha=0.05 here. Its
# symmetric bounds use the 95th Student-t percentile for each support shift.
valid_result: vl.ValidResult = vl.ValidEngine(
    [vl.MetricMAVM(alpha=0.05, duplicates=vl.EMAVMDuplicates.REJECT)]
).run(valid_case)
print("Simulation:", sim_data.shape, "(channel, epis_trans, aleatory)")
print(pd.DataFrame(channel_status))
print("TC3 result axes:", valid_result.metrics["TC3"][vl.EMetricType.MAVM].axes)

# %%
# 3. Post-processing and inspect data structures
# ----------------------------------------------
# The paper script picks WHOLE scenarios with the smallest/largest sum of ECDF
# support values. This is different from pointwise envelope bounds when CDFs
# cross. The nominal curve is the average of the conditional quantile functions.
# Both bound rules are available in PyVale; use SUPPORT_SUM for literal parity.
metric_rows = [
    f"sim-{sim_name}_exp-{exp_name}-{direction}"
    for exp_name in ("min", "max")
    for sim_name in ("min", "max")
    for direction in ("d+", "d-")
]
metric_table: dict[str, list[float]] = {}
for channel in measurements:
    units = observables[channel].phys_units
    axis_label = "Coil RMS voltage [V]" if channel == "CV" else "Temperature [degC]"
    model_scenarios: list[np.ndarray] = vl.extract_aleatory_scenarios(
        sim_arrays[channel]
    )
    model_summary: vl.ValidECDFSummary = vl.calc_ecdf_summary(
        model_scenarios, bounds=vl.EValidECDFBounds.SUPPORT_SUM
    )
    # %%
    # 4. Analysis and visualisation
    # ------------------------------
    # Family plots retain the 250 conditional model ECDFs for every channel.
    # Available experimental readings retain their individual measurement spread.
    labels = vl.ValidPlotOpts(title=channel, x_label=axis_label)
    fig: Figure
    ax: Axes
    ecdf_plot: vl.ECDFPlot = vl.plot_sim_ecdfs(
        valid_case,
        channel,
        valid_plot_opts=labels,
        view=vl.ValidECDFView(
            scenario_idxs=None, centre=vl.EValidECDFCentre.MEAN
        ),
    )
    fig, axes = ecdf_plot.figure, ecdf_plot.axes
    figure_name = f"0_sim_allcdfs_{channel}.png"
    fig.savefig(output_path / figure_name, bbox_inches="tight")
    if not SHOW_PLOTS:
        plt.close(fig)
    if np.any(np.isnan(exp_arrays[channel].data)):
        continue

    supports = vl.find_metric_epis_supports(
        valid_case, valid_result, channel, vl.EMetricType.MAVM
    )
    for metric_output, direction in (
        (vl.EMetricOut.D_PLUS, "d_plus"),
        (vl.EMetricOut.D_MINUS, "d_minus"),
    ):
        limiting_plot = vl.plot_sim_exp_cdfs(valid_case, channel, valid_plot_opts=labels)
        vl.add_support_overlay(limiting_plot, supports, output=metric_output)
        limiting_plot.figure.savefig(
            output_path / f"mavm_{direction}_limiting_ecdf_{channel}.png",
            bbox_inches="tight",
        )
        if not SHOW_PLOTS:
            plt.close(limiting_plot.figure)

    exp_scenarios: list[np.ndarray] = vl.extract_aleatory_scenarios(
        exp_arrays[channel], offset=None
    )
    exp_summary: vl.ValidECDFSummary = vl.calc_ecdf_summary(
        exp_scenarios,
        bounds=vl.EValidECDFBounds.SUPPORT_SUM,
    )
    model_lower = model_scenarios[cast(int, model_summary.lower_idx)]
    model_upper = model_scenarios[cast(int, model_summary.upper_idx)]
    exp_lower: np.ndarray = exp_arrays[channel].select_aleatory(
        0, 0, 0, offset=vl.EEpisOffset.LOWER
    )
    exp_upper: np.ndarray = exp_arrays[channel].select_aleatory(
        0, 0, 0, offset=vl.EEpisOffset.UPPER
    )
    correction: vl.MAVMCorrection = vl.calc_mavm_correction(
        model_lower,
        model_upper,
        exp_lower,
        exp_upper,
        duplicates=vl.EMAVMDuplicates.REJECT,
    )

    # %%
    # Export the original four endpoint comparisons using the engine arrays.
    # These are distinct from the worst MAVM over every retained scenario.
    metric_field: vl.MetricField = valid_result.metrics[channel][
        vl.EMetricType.MAVM
    ]
    distances = []
    for offset in (vl.EEpisOffset.LOWER, vl.EEpisOffset.UPPER):
        for model_idx in (
            cast(int, model_summary.lower_idx),
            cast(int, model_summary.upper_idx),
        ):
            distances.extend(
                [
                    float(
                        metric_field.outputs[vl.EMetricOut.D_PLUS][
                            0, 0, 0, model_idx, offset
                        ]
                    ),
                    float(
                        metric_field.outputs[vl.EMetricOut.D_MINUS][
                            0, 0, 0, model_idx, offset
                        ]
                    ),
                ]
            )
    metric_table[channel] = distances

    # %%
    # The next four figures reproduce the diagnostic families in the reference:
    # source distributions; endpoint correction cases; the outer correction with
    # experiment visible; and model-only uncertainty/correction for a clean view.
    # Gray fill shows the MAVM-corrected model support envelope.
    diagnostic_views = (
        ("1_exp-sim-cdfs", vl.EValidMAVMDiagnostic.DISTRIBUTIONS),
        ("2_dextremes_wcdfs", vl.EValidMAVMDiagnostic.COMPARISONS),
        ("3_dextremes_wexpcdfs", vl.EValidMAVMDiagnostic.CORRECTION),
        ("4_dextremes_simcdfsonly", vl.EValidMAVMDiagnostic.MODEL_CORRECTION),
    )
    for figure_prefix, view in diagnostic_views:
        fig, ax = vl.plot_valid_mavm_diagnostic(
            model_summary,
            exp_summary,
            correction=correction,
            view=view,
            valid_plot_opts=labels,
        )
        figure_name = f"{figure_prefix}_{channel}.png"
        fig.savefig(output_path / figure_name, bbox_inches="tight")
        if not SHOW_PLOTS:
            plt.close(fig)

# %%
calculated = pd.DataFrame(metric_table, index=metric_rows)
calculated.to_csv(output_path / "pointsensors_mavm.csv")
print(calculated)
if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
