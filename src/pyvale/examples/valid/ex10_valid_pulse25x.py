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
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import pyvale.data as dataset
from pyvale import valid
from pyvale.valid import validplot

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
with (data_path / "pulse25x_provenance.json").open() as stream:
    provenance = json.load(stream)

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
observables: dict[str, valid.ValidObservable] = {}
sim_arrays: dict[str, valid.ValidArray] = {}
exp_arrays: dict[str, valid.ValidArray] = {}
channel_status = []
for index, (channel, observations) in enumerate(measurements.items()):
    units = "V" if channel == "CV" else "degC"
    observables[channel] = valid.ValidObservable(
        components=(channel,),
        geometry=valid.EValidGeometry.GLOBAL,
        phys_units=units,
    )
    sim_arrays[channel] = valid.ValidArray(
        sim_data[sensor_tags.index(channel)],
        (valid.EValidAxis.EPISTEMIC_TRANS, valid.EValidAxis.ALEATORY),
    )
    missing = bool(np.any(np.isnan(observations)))
    # TC2 and CV have repeated readings. The legacy integration discards their
    # multiplicities. Explicit unique samples reproduce that weighting; ordinary
    # PyVale INCLUDE mode otherwise retains empirical duplicate probability mass.
    exp_support = np.array([np.nan]) if missing else np.unique(observations)
    magnitude = uncertainties[index]
    exp_arrays[channel] = valid.ValidArray(
        exp_support,
        (valid.EValidAxis.ALEATORY,),
        offsets=valid.EpistemicOffsets(lower=magnitude, upper=magnitude),
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
valid_case = valid.ValidCase(observables, sim_arrays, exp_arrays)
pd.DataFrame(channel_status).to_csv(output_path / "channel_status.csv", index=False)

# %%
# 2. Run validation analysis
# --------------------------
# Keep every epistemic setting and all three experimental offset positions.
# For each scalar channel the result shape is (point=1, component=1, time=1,
# sim_trans=250, exp_offset=3). Missing TC1, TC4 and TC7 retain NaN results.
# The reference uses t.ppf(0.95, n_unique-1), matching alpha=0.05 here. Its
# symmetric bounds use the 95th Student-t percentile for each support shift.
valid_result: valid.ValidResult = valid.ValidEngine(
    [valid.MetricMAVM(alpha=0.05, duplicates=valid.EMAVMDuplicates.REJECT)]
).run(valid_case)
print("Simulation:", sim_data.shape, "(channel, epis_trans, aleatory)")
print(pd.DataFrame(channel_status))
print("TC3 result axes:", valid_result.metrics["TC3"][valid.EMetricType.MAVM].axes)

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
selected_scenarios: dict[str, dict[str, int]] = {}
axis_limits: dict[str, tuple[float, float]] = {}
for channel in measurements:
    units = observables[channel].phys_units
    axis_label = "Coil RMS voltage [V]" if channel == "CV" else "Temperature [degC]"
    model_scenarios: list[np.ndarray] = valid.extract_aleatory_scenarios(
        sim_arrays[channel]
    )
    model_summary: valid.ValidECDFSummary = valid.calc_ecdf_summary(
        model_scenarios, bounds=valid.EValidECDFBounds.SUPPORT_SUM
    )
    selected_scenarios[channel] = {
        "lower": cast(int, model_summary.lower_idx),
        "upper": cast(int, model_summary.upper_idx),
    }

    # %%
    # 4. Analysis and visualisation
    # ------------------------------
    # Family plots retain the 250 conditional model ECDFs for every channel.
    # Available experimental readings retain their individual measurement spread.
    labels = validplot.ValidPlotOpts(title=channel, x_label=axis_label)
    fig: Figure
    ax: Axes
    fig, axes = validplot.plot_sim_ecdfs(
        valid_case,
        channel,
        valid_plot_opts=labels,
        view=valid.ValidECDFView(
            scenario_idxs=None, centre=valid.EValidECDFCentre.MEAN
        ),
    )
    figure_name = f"0_sim_allcdfs_{channel}.png"
    fig.savefig(output_path / figure_name, bbox_inches="tight")
    axis_limits[figure_name] = axes[0, 0].get_xlim()
    if not SHOW_PLOTS:
        plt.close(fig)
    if np.any(np.isnan(exp_arrays[channel].data)):
        continue

    exp_scenarios: list[np.ndarray] = valid.extract_aleatory_scenarios(
        exp_arrays[channel], offset=None
    )
    exp_summary: valid.ValidECDFSummary = valid.calc_ecdf_summary(
        exp_scenarios,
        bounds=valid.EValidECDFBounds.SUPPORT_SUM,
    )
    model_lower = model_scenarios[cast(int, model_summary.lower_idx)]
    model_upper = model_scenarios[cast(int, model_summary.upper_idx)]
    exp_lower: np.ndarray = exp_arrays[channel].select_aleatory(
        0, 0, 0, offset=valid.EEpisOffset.LOWER
    )
    exp_upper: np.ndarray = exp_arrays[channel].select_aleatory(
        0, 0, 0, offset=valid.EEpisOffset.UPPER
    )
    correction: valid.MAVMCorrection = valid.calc_mavm_correction(
        model_lower,
        model_upper,
        exp_lower,
        exp_upper,
        duplicates=valid.EMAVMDuplicates.REJECT,
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
    metric_field: valid.ValidMetricField = valid_result.metrics[channel][
        valid.EMetricType.MAVM
    ]
    distances = []
    for offset in (valid.EEpisOffset.LOWER, valid.EEpisOffset.UPPER):
        for model_idx in (
            cast(int, model_summary.lower_idx),
            cast(int, model_summary.upper_idx),
        ):
            distances.extend(
                [
                    float(
                        metric_field.outputs[valid.EMetricOutput.D_PLUS][
                            0, 0, 0, model_idx, offset
                        ]
                    ),
                    float(
                        metric_field.outputs[valid.EMetricOutput.D_MINUS][
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
        ("1_exp-sim-cdfs", valid.EValidMAVMDiagnostic.DISTRIBUTIONS),
        ("2_dextremes_wcdfs", valid.EValidMAVMDiagnostic.COMPARISONS),
        ("3_dextremes_wexpcdfs", valid.EValidMAVMDiagnostic.CORRECTION),
        ("4_dextremes_simcdfsonly", valid.EValidMAVMDiagnostic.MODEL_CORRECTION),
    )
    for figure_prefix, view in diagnostic_views:
        fig, ax = validplot.plot_valid_mavm_diagnostic(
            model_summary,
            exp_summary,
            correction=correction,
            view=view,
            valid_plot_opts=labels,
        )
        figure_name = f"{figure_prefix}_{channel}.png"
        fig.savefig(output_path / figure_name, bbox_inches="tight")
        axis_limits[figure_name] = axes[0, 0].get_xlim()
        if not SHOW_PLOTS:
            plt.close(fig)

# %%
# Check every d+ and d- against the independently executed legacy script's CSV:
# eight available channels × four endpoint pairs × two directions = 64 values.
# The frozen reference and its source hashes are bundled with the full dataset.
calculated = pd.DataFrame(metric_table, index=metric_rows)
reference: pd.DataFrame = pd.read_csv(
    data_path / "pulse25x_reference_mavm.csv", index_col=0
)
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
print(calculated)
print(parity)
if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
