# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Load experimental sensor data
================================================================================

Use a point-sensor loader for each data acquisition system to read the
Pulse253–255 measurements into named experimental data containers.
"""

from pathlib import Path

import pyvale.data as dataset
import pyvale.dataio as io

# %%
data_path: Path = dataset.valid_data_dir()
pulse_ids = (253, 254, 255)
load_opts = io.ExpLoadOpts(delimiter=",", header_rows=0)

exp_loaders: dict[str, io.IExpLoader] = {
    "DIC-DAQ": io.PointSensLoader(
        load_files=[data_path / f"Pulse{pp}_SteadyDICData.csv" for pp in pulse_ids],
        sens_cols=(2, 3, 4, 5, 6, 7, 8, 9, 10),
        sens_labels=("TC1", "TC3", "TC4", "TC5", "TC6", "TC7", "TC8", "TC9", "TC10"),
        load_opts=load_opts,
        time_col=1,
    ),
    "HIVE-DAQ": io.PointSensLoader(
        load_files=[data_path / f"Pulse{pp}_SteadyHIVEData.csv" for pp in pulse_ids],
        sens_cols=(2,),
        sens_labels=("TC2",),
        load_opts=load_opts,
        time_col=1,
    ),
    "PICO-DAQ": io.PointSensLoader(
        load_files=[data_path / f"Pulse{pp}_SteadyPICOData.csv" for pp in pulse_ids],
        sens_cols=(1,),
        sens_labels=("CV",),
        load_opts=load_opts,
    ),
}

# %%
exp_data: dict[str, io.ExpData] = io.load_exp_data(loaders=exp_loaders)
for daq_name, acquisition in exp_data.items():
    print(f"{daq_name}: {acquisition.fields.shape} (sensors, samples)")
    print(f"  Sensors: {tuple(acquisition.sens_label_to_ind)}")

dic_data = exp_data["DIC-DAQ"]
tc3 = dic_data.fields[dic_data.sens_label_to_ind["TC3"]]
print("First 5 TC3 measurements [degC]:", tc3[:5])
