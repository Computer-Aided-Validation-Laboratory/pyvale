# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Example 5: Comparative Benchmarking of Validation Metrics.

Demonstrates computing and comparing multiple validation metrics (MAVM,
classical AVM, Kolmogorov-Smirnov, Cramér-von Mises, RMSE) across point
sensors using the IValMetric strategy pattern.
"""

from pathlib import Path
import numpy as np

import pyvale.dataset as dataset
import pyvale.dataio as io
import pyvale.valid as val


def main() -> None:
    data_path = dataset.valid_data_dir()
    load_opts = io.ExpLoadOpts(delimiter=",", header_rows=0)

    # 1. Load experimental data
    exp_loaders = {
        "DIC-DAQ": io.PointSensLoader(
            load_files=[
                data_path / "Pulse253_SteadyDICData.csv",
                data_path / "Pulse254_SteadyDICData.csv",
                data_path / "Pulse255_SteadyDICData.csv",
            ],
            sens_cols=np.array([3, 5, 6]),
            sens_labels=["TC3", "TC5", "TC6"],
            load_opts=load_opts,
        ),
        "HIVE-DAQ": io.PointSensLoader(
            load_files=[
                data_path / "Pulse253_SteadyHIVEData.csv",
                data_path / "Pulse254_SteadyHIVEData.csv",
                data_path / "Pulse255_SteadyHIVEData.csv",
            ],
            sens_cols=np.array([2]),
            sens_labels=["TC2"],
            load_opts=load_opts,
        ),
    }
    exp_data = io.load_exp_data(exp_loaders)
    exp_val_data = val.extract_val_data_by_key(exp_data)

    # 2. Load simulation data
    sim_csv = data_path / "SamplingResultsOnlyPointSensors.csv"
    sens_keys = {"TC2": 1, "TC3": 2, "TC5": 4, "TC6": 5}
    sim_val_data = val.load_prob_sim_csv(
        csv_path=sim_csv,
        sens_keys=sens_keys,
        n_epistemic=50,
        n_aleatory=100,
    )

    # 3. Define metrics to evaluate via strategy pattern
    metrics: dict[str, val.IValMetric] = {
        "MAVM": val.MetricMAVM(alpha=0.05),
        "AVM (W1)": val.MetricAVM(),
        "KS (L_inf)": val.MetricKS(),
        "CvM (L2)": val.MetricCVM(),
        "RMSE": val.MetricRMSE(),
    }

    # 4. Evaluate each metric across all sensors
    results: dict[str, dict[str, float]] = {}
    for metric_name, strategy in metrics.items():
        raw_res = val.calc_metric_point(strategy, sim_val_data, exp_val_data)
        results[metric_name] = {}
        for sens_lbl, val_res in raw_res.items():
            if isinstance(val_res, val.MAVMResult):
                results[metric_name][sens_lbl] = val_res.d_total
            else:
                results[metric_name][sens_lbl] = float(val_res)

    # 5. Print formatted comparison table
    print(80 * "=")
    print("Validation Metrics Comparison Across Thermocouples:")
    print(80 * "=")
    sensor_list = list(next(iter(results.values())).keys())
    hdr = f"{'Metric':<14}" + "".join(f"{s:<12}" for s in sensor_list)
    print(hdr)
    print(80 * "-")
    for metric_name, sens_dict in results.items():
        row = f"{metric_name:<14}" + "".join(
            f"{sens_dict[s]:<12.3f}" for s in sensor_list
        )
        print(row)
    print(80 * "=")


if __name__ == "__main__":
    main()
