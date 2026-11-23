# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Pulse25X benchmark
================================================================================
"""

from itertools import count
from pathlib import Path
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pyvale.data as dataset
import pyvale.validation as vl

matplotlib.use("Agg")

# %%
SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")
output_path = Path.cwd() / "pyvale-output" / "valid_ex10_pulse25x_benchmark"
output_path.mkdir(parents=True, exist_ok=True)
figure_numbers = count(1)

# %%
data_path: Path = dataset.valid_data_dir()
with np.load(data_path / "pulse25x_full_simulation.npz") as saved:
    sensor_tags = tuple(saved["sensor_tags"].tolist())
    sim_data: np.ndarray = saved["simulation"]
# %%
measurements: dict[str, np.ndarray] = {}
exp_sources = (
    ("SteadyDICData", ("TC1", "TC3", "TC4", "TC5", "TC6", "TC7", "TC8", "TC9", "TC10")),
    ("SteadyHIVEData", ("TC2",)),
    ("SteadyPICOData", ("CV",)),
)
for ss, cc in exp_sources:
    measurement_table: pd.DataFrame = pd.read_csv(data_path / f"Pulse255_{ss}.csv")
    for kk in cc:
        column = "Coil RMS Voltage" if kk == "CV" else kk
        measurements[kk] = measurement_table[column].to_numpy(dtype=np.float64)

# %%
summary: pd.DataFrame = pd.read_csv(data_path / "pulse25x_uncertainty_summary.csv")
uncertainty_columns = (*sensor_tags[:-1], "Coil RMS Voltage")
uncertainties = summary.loc[summary.index[-1], list(uncertainty_columns)].to_numpy(
    dtype=float
)
observables: dict[str, vl.ValidObservable] = {}
sim_arrays: dict[str, vl.ValidArray] = {}
exp_arrays: dict[str, vl.ValidArray] = {}
for ii, (kk, oo) in enumerate(measurements.items()):
    units = "V" if kk == "CV" else "degC"
    observables[kk] = vl.ValidObservable(
        components=(kk,),
        geometry=vl.EValidGeometry.GLOBAL,
        phys_units=units,
    )
    sim_arrays[kk] = vl.ValidArray(
        sim_data[sensor_tags.index(kk)],
        (vl.EValidAxis.EPISTEMIC_TRANS, vl.EValidAxis.ALEATORY),
    )
    missing = bool(np.any(np.isnan(oo)))
    exp_support = np.array([np.nan]) if missing else np.unique(oo)
    magnitude = uncertainties[ii]
    exp_arrays[kk] = vl.ValidArray(
        exp_support,
        (vl.EValidAxis.ALEATORY,),
        offsets=vl.EpistemicOffsets(lower=magnitude, upper=magnitude),
    )
valid_case = vl.ValidCase(observables, sim_arrays, exp_arrays)

# %%
valid_result: vl.ValidResult = vl.ValidEngine(
    [vl.MetricMAVM(alpha=0.05, duplicates=vl.EMAVMDuplicates.REJECT)]
).run(valid_case)
print("Simulation:", sim_data.shape, "(channel, epis_trans, aleatory)")
print("TC3 result axes:", valid_result.metrics["TC3"][vl.EMetricType.MAVM].axes)

# %%
for kk in measurements:
    units = observables[kk].phys_units
    axis_label = "Coil RMS voltage [V]" if kk == "CV" else "Temperature [degC]"
    model_scenarios: list[np.ndarray] = vl.extract_aleatory_scenarios(
        sim_arrays[kk]
    )
    model_summary: vl.ValidECDFSummary = vl.calc_ecdf_summary(
        model_scenarios, bounds=vl.EValidECDFBounds.SUPPORT_SUM
    )
    # %%
    labels = vl.ValidPlotOpts(title=kk, x_label=axis_label)
    ecdf_plot: vl.ECDFPlot = vl.plot_sim_ecdfs(
        valid_case,
        kk,
        valid_plot_opts=labels,
        view=vl.ValidECDFView(
            scenario_idxs=None, centre=vl.EValidECDFCentre.MEAN
        ),
    )
    figure_name = f"val_ex10_fig{next(figure_numbers):02d}_0_sim_allcdfs_{kk}.png"
    ecdf_plot.figure.savefig(output_path / figure_name, bbox_inches="tight")
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
    limiting_plot.figure.savefig(
        output_path
        / (
            f"val_ex10_fig{next(figure_numbers):02d}_"
            f"mavm_support_envelope_{kk}.png"
        ),
        bbox_inches="tight",
    )
    if not SHOW_PLOTS:
        plt.close(limiting_plot.figure)

    exp_scenarios: list[np.ndarray] = vl.extract_aleatory_scenarios(
        exp_arrays[kk], offset=None
    )
    exp_summary: vl.ValidECDFSummary = vl.calc_ecdf_summary(
        exp_scenarios,
        bounds=vl.EValidECDFBounds.SUPPORT_SUM,
    )
    model_lower = model_scenarios[cast(int, model_summary.lower_idx)]
    model_upper = model_scenarios[cast(int, model_summary.upper_idx)]
    exp_lower: np.ndarray = exp_arrays[kk].select_aleatory(
        0, 0, 0, offset=vl.EEpisOffset.LOWER
    )
    exp_upper: np.ndarray = exp_arrays[kk].select_aleatory(
        0, 0, 0, offset=vl.EEpisOffset.UPPER
    )
    correction: vl.MAVMBoundPairs = vl.calc_mavm_bound_pairs(
        model_lower,
        model_upper,
        exp_lower,
        exp_upper,
        duplicates=vl.EMAVMDuplicates.REJECT,
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
            model_summary,
            exp_summary,
            correction=correction,
            view=vv,
            valid_plot_opts=labels,
        )
        figure_name = f"val_ex10_fig{next(figure_numbers):02d}_{ff}_{kk}.png"
        fig.savefig(output_path / figure_name, bbox_inches="tight")
        if not SHOW_PLOTS:
            plt.close(fig)

if SHOW_PLOTS:
    plt.show()
else:
    plt.close("all")
