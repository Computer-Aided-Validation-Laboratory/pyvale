# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Pulse25X benchmark
================================================================================

Reproduce the Pulse25X benchmark and inspect the MAVM endpoint and support
plots.
"""

from itertools import count
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.validation as vl

matplotlib.use("Agg")

# %%
SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")
output_path = Path.cwd() / "pyvale-output" / "valid_ex10_pulse25x_benchmark"
output_path.mkdir(parents=True, exist_ok=True)
fig_nums = count(1)

# %%
# 1. Load simulation samples
# --------------------------
#
# The archive stores one array ordered by channel, transferable case, and
# aleatory sample. The channel tags identify which slice to use for each
# observable.
data_path: Path = dataset.valid_data_dir()
with np.load(data_path / "pulse25x_full_simulation.npz") as saved:
    sensor_tags = tuple(saved["sensor_tags"].tolist())
    sim_data: np.ndarray = saved["simulation"]

# %%
# 2. Read the three acquisition streams
# -------------------------------------
#
# Each ``PointSensLoader`` maps CSV columns to sensor labels. The returned
# ``ExpData`` objects keep one row per channel and one column per recorded
# sample.
dic_tags = ("TC1", "TC3", "TC4", "TC5", "TC6", "TC7", "TC8", "TC9", "TC10")
exp_loaders: dict[str, io.IExpLoader] = {
    "DIC_DAQ": io.PointSensLoader(
        load_files=data_path / "Pulse255_SteadyDICData.csv",
        sens_cols=tuple(range(2, 11)),
        sens_labels=dic_tags,
    ),
    "HIVE_DAQ": io.PointSensLoader(
        load_files=data_path / "Pulse255_SteadyHIVEData.csv",
        sens_cols=(2,),
        sens_labels=("TC2",),
    ),
    "PICO_DAQ": io.PointSensLoader(
        load_files=data_path / "Pulse255_SteadyPICOData.csv",
        sens_cols=(1,),
        sens_labels=("CV",),
    ),
}
exp_data = io.load_exp_data(loaders=exp_loaders)


# %%
# 3. Read the uncertainty summary
# -------------------------------
#
# The benchmark uses the final row of the summary to construct symmetric
# offset magnitudes. Keep the benchmark's positional mapping when reproducing
# its reference results.
summary: pd.DataFrame = pd.read_csv(data_path / "pulse25x_uncertainty_summary.csv")
uncertainty_columns = (*sensor_tags[:-1], "Coil RMS Voltage")
uncertainties = summary.loc[summary.index[-1], list(uncertainty_columns)].to_numpy(
    dtype=float
)

# %%
# 4. Assemble channels for validation
# -----------------------------------
#
# Map each validation tag to a DAQ and sensor label. A ``POINT`` observable
# can represent an individually identified sensor with optional coordinates.
# The steady window turns each channel trace into a set of aleatory samples.
channel_map = {kk: ("DIC_DAQ", kk) for kk in dic_tags}
channel_map["TC2"] = ("HIVE_DAQ", "TC2")
channel_map["CV"] = ("PICO_DAQ", "CV")
observables: dict[str, vl.ValidObservable] = {}
for kk in channel_map:
    observables[kk] = vl.ValidObservable(
        components=(kk,),
        geometry=vl.EValidGeometry.POINT,
        phys_units="V" if kk == "CV" else "degC",
    )

sim_axes = (vl.EValidAxis.EPISTEMIC_TRANS, vl.EValidAxis.ALEATORY)
sim_arrays: dict[str, vl.ValidArray] = {}
for kk in observables:
    channel_samples = sim_data[sensor_tags.index(kk)]
    sim_arrays[kk] = vl.ValidArray(channel_samples, sim_axes)

exp_arrays = vl.assemble_exp_arrays(
    sources=exp_data,
    observables=observables,
    steady_window=slice(None),
    channel_map=channel_map,
)

# %%
# 5. Apply the benchmark sample rules
# -----------------------------------
#
# Use unique observed values for the benchmark integration and mark channels
# containing missing values as unavailable. Apply the uncertainty vector in
# DAQ channel order to match the frozen reference.
for ii, (kk, aa) in enumerate(exp_arrays.items()):
    oo = aa.select_aleatory(0, 0, 0).astype(np.float64, copy=False)
    missing = bool(np.any(np.isnan(oo)))
    exp_support = np.array([np.nan]) if missing else np.unique(oo)
    magnitude = uncertainties[ii]

    exp_arrays[kk] = vl.ValidArray(
        exp_support,
        (vl.EValidAxis.ALEATORY,),
        offsets=vl.EpistemicOffsets(lower=magnitude, upper=magnitude),
    )



valid_case = vl.ValidCase(observables, sim_arrays, exp_arrays)

valid_result: vl.ValidResult = vl.ValidEngine(
    [vl.MetricMAVM(alpha=0.05, duplicates=vl.EMAVMDupes.REJECT)]
).run(valid_case)

print(80*"-")
print("Simulation:", sim_data.shape, "(channel, epis_trans, aleatory)")
print("TC3 result axes:", valid_result.metrics["TC3"][vl.EMetricType.MAVM].axes)
print(80*"-")

# %%
# 6. Inspect the support and endpoint comparisons
# -----------------------------------------------
#
# The support-sum rule selects complete endpoint ECDFs by summing their sorted
# support values. ``calc_mavm_bound_pairs`` compares the simulation and
# experiment endpoints and returns the pairs used to shift the model support.
# Choose a diagnostic view to inspect the distributions, comparisons, or
# shifted support.
for kk in observables:
    units = observables[kk].phys_units
    axis_label = "Coil RMS voltage [V]" if kk == "CV" else "Temperature [degC]"
    sim_view = vl.calc_ecdf_view(sim_arrays[kk])
    sim_cases = vl.select_ecdf_support_sum_cases(sim_view.samples)

    labels = vl.ValidPlotOpts(title=kk, x_label=axis_label)
    ecdf_plot: vl.ECDFPlot = vl.plot_sim_ecdfs(
        valid_case,
        kk,
        valid_plot_opts=labels,
        vis_opts=vl.ECDFVisOpts(
            epis_case_idxs=None, centre=vl.EECDFCentre.MEAN
        ),
    )

    fig_file = f"val_ex10_fig{next(fig_nums):02d}_0_sim_allcdfs_{kk}.png"
    ecdf_plot.figure.savefig(output_path / fig_file, bbox_inches="tight")

    if not SHOW_PLOTS:
        plt.close(ecdf_plot.figure)

    if np.any(np.isnan(exp_arrays[kk].data)):
        continue

    supports = vl.find_metric_epis_supports(
        valid_case,
        valid_result,
        kk,
        vl.EMetricType.MAVM,
        selectors={vl.EValidResAxis.EXP_OFFSET: vl.EEpisOffset.NOMINAL},
    )
    limiting_plot = vl.plot_sim_exp_cdfs(valid_case, kk, valid_plot_opts=labels)
    vl.add_support_overlay(limiting_plot, supports)

    fig_file = f"val_ex10_fig{next(fig_nums):02d}_mavm_support_envelope_{kk}.png"
    limiting_plot.figure.savefig(output_path / fig_file, bbox_inches="tight")

    if not SHOW_PLOTS:
        plt.close(limiting_plot.figure)

    exp_view = vl.calc_ecdf_view(exp_arrays[kk], offset=None)
    exp_cases = vl.select_ecdf_support_sum_cases(exp_view.samples)

    sim_lower = sim_view.samples[sim_cases[0]]
    sim_upper = sim_view.samples[sim_cases[1]]

    exp_lower: np.ndarray = exp_arrays[kk].select_aleatory(
        0, 0, 0, offset=vl.EEpisOffset.LOWER
    )
    exp_upper: np.ndarray = exp_arrays[kk].select_aleatory(
        0, 0, 0, offset=vl.EEpisOffset.UPPER
    )

    bound_pairs: vl.MAVMBoundPairs = vl.calc_mavm_bound_pairs(
        sim_lower,
        sim_upper,
        exp_lower,
        exp_upper,
        duplicates=vl.EMAVMDupes.REJECT,
    )

    # %%
    diagnostic_views = (
        ("1_exp-sim-cdfs", vl.EValidMAVMDiagnostic.DISTRIBUTIONS),
        ("2_dextremes_wcdfs", vl.EValidMAVMDiagnostic.COMPARISONS),
        ("3_dextremes_wexpcdfs", vl.EValidMAVMDiagnostic.CORRECTION),
        ("4_dextremes_simcdfsonly", vl.EValidMAVMDiagnostic.MODEL_CORRECTION),
    )
    for ff, vv in diagnostic_views:
        fig, ax = vl.plot_valid_mavm_diagnostic(
            sim_view,
            exp_view,
            sim_support_cases=sim_cases,
            exp_support_cases=exp_cases,
            bound_pairs=bound_pairs,
            view=vv,
            valid_plot_opts=labels,
        )

        fig_file = f"val_ex10_fig{next(fig_nums):02d}_{ff}_{kk}.png"
        fig.savefig(output_path / fig_file, bbox_inches="tight")

        if not SHOW_PLOTS:
            plt.close(fig)

# %%
# TC3 results
# -----------
#
# These figures show the benchmark diagnostics for thermocouple TC3.
# The script saves the available figures for every channel.
#
# .. image:: /_static/val_ex10_fig03_mavm_support_envelope_TC3.png
#    :alt: TC3 simulation and experiment ECDFs with the MAVM support envelope
#    :width: 100%
#    :align: center
#
# .. image:: /_static/val_ex10_fig05_2_dextremes_wcdfs_TC3.png
#    :alt: TC3 shifted endpoint curves for the paired MAVM comparisons
#    :width: 100%
#    :align: center

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
