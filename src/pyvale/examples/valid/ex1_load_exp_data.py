# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Example 1: Loading Experimental Point Sensor Data from Multiple DAQs.

Demonstrates using PointSensLoader to load time-series measurements from
multiple DAQ files (DIC-DAQ, HIVE-DAQ, PICO-DAQ), extracting sensor traces,
and structuring them into ExpData and PointValData containers.
"""

from pathlib import Path
import numpy as np

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.valid as val


def main() -> None:
    data_path = dataset.valid_data_dir()
    load_opts = io.ExpLoadOpts(delimiter=",", header_rows=0)

    # Configure loaders for multiple DAQ devices
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
            time_col=1,
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
            time_col=1,
        ),
        "PICO-DAQ": io.PointSensLoader(
            load_files=[
                data_path / "Pulse253_SteadyPICOData.csv",
                data_path / "Pulse254_SteadyPICOData.csv",
                data_path / "Pulse255_SteadyPICOData.csv",
            ],
            sens_cols=np.array([1]),
            sens_labels=["CV"],
            load_opts=load_opts,
            time_col=None,
        ),
    }

    # Load experimental data into ExpData containers
    exp_data: dict[str, io.ExpData] = io.load_exp_data(exp_loaders)

    print(80 * "=")
    print("Loaded Experimental Data Summary:")
    print(80 * "=")
    for daq_name, daq_exp in exp_data.items():
        print(f"DAQ: {daq_name}")
        print(f"  Fields shape: {daq_exp.fields.shape}")
        print(f"  Sensors: {list(daq_exp.sens_label_to_ind.keys())}")
        if daq_exp.times is not None:
            print(f"  Time steps: {len(daq_exp.times)}")
        print()

    # Extract into PointValData for validation analysis
    val_data: val.PointValData = val.extract_val_data_by_key(
        exp_data=exp_data,
        sensor_keys={
            "DIC-DAQ": ["TC3", "TC5", "TC6"],
            "HIVE-DAQ": ["TC2"],
        },
        steady_slice=slice(0, 50),
    )

    print("PointValData Extraction:")
    for daq_k, arr in val_data.val_points.items():
        print(f"  {daq_k} shape: {arr.shape}")


if __name__ == "__main__":
    main()
