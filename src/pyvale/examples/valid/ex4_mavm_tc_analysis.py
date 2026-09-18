# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Example 4: Full MAVM Analysis of Point Sensors (Thermocouples).

Demonstrates an end-to-end model validation workflow comparing experimental
thermocouple measurements (across multiple test pulses) against probabilistic
finite element thermal simulations using the Modified Area Validation Metric.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import pyvale.dataset as dataset
import pyvale.dataio as io
import pyvale.valid as val


def main() -> None:
    data_path = dataset.valid_data_dir()
    load_opts = io.ExpLoadOpts(delimiter=",", header_rows=0)

    # 1. Load experimental thermocouple traces across test pulses 253, 254, 255
    exp_loaders = {
        "DIC-DAQ": io.PointSensLoader(
            load_files=[
                data_path / "Pulse253_SteadyDICData.csv",
                data_path / "Pulse254_SteadyDICData.csv",
                data_path / "Pulse255_SteadyDICData.csv",
            ],
            sens_cols=np.array([3, 5, 6, 8, 9, 10]),
            sens_labels=["TC3", "TC5", "TC6", "TC8", "TC9", "TC10"],
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

    # 2. Load probabilistic simulation samples
    sim_csv = data_path / "SamplingResultsOnlyPointSensors.csv"
    sens_keys = {
        "TC2": 1,
        "TC3": 2,
        "TC5": 4,
        "TC6": 5,
        "TC8": 7,
        "TC9": 8,
        "TC10": 9,
    }
    sim_val_data = val.load_prob_sim_csv(
        csv_path=sim_csv,
        sens_keys=sens_keys,
        n_epistemic=50,
        n_aleatory=100,
    )

    # 3. Compute MAVM for all thermocouples
    mavm_results = val.calc_mavm_point(
        sim_data=sim_val_data,
        exp_data=exp_val_data,
        alpha=0.05,
    )

    print(80 * "=")
    print("Modified Area Validation Metric (MAVM) Results:")
    print(80 * "=")
    hdr = f"{'Sensor':<8} {'d+ (Over)':<14} {'d- (Under)':<14} {'d_total':<14}"
    print(hdr)
    print(80 * "-")
    for lbl, res in mavm_results.items():
        print(
            f"{lbl:<10} {res.d_plus:<15.3f} {res.d_minus:<15.3f} "
            f"{res.d_total:<15.3f}"
        )
    print(80 * "=")

    # 4. Plot individual MAVM CDF comparison for TC3
    fig_cdf, ax_cdf = val.plot_mavm_cdf_1d(
        mavm_results["TC3"],
        title="Thermocouple TC3 MAVM Validation",
        unit=r"^\circ\text{C}",
    )
    fig_cdf.savefig("ex4_mavm_tc3_cdf.png", dpi=200)

    # 5. Plot summary bar chart across all thermocouples
    fig_bar, ax_bar = val.plot_mavm_summary_bars(
        mavm_results,
        title="Thermocouple Point Sensors: MAVM Validation Summary",
        unit=r"^\circ\text{C}",
    )
    fig_bar.savefig("ex4_mavm_summary_bars.png", dpi=200)

    print("\nSaved visualization plots to current directory.")


if __name__ == "__main__":
    main()
