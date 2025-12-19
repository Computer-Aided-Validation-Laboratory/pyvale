# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2024 The Computer Aided Validation Team
# ==============================================================================

from pathlib import Path
import pyvale.dataset as dataset
import pyvale.dataio as io
import pyvale.valid as val


data_path = dataset.valid_data_dir()
print(f"{data_path}")

val_csvs = dataset.valid_data_csvs()
print("Validation data csvs:")
for ff in val_csvs:
    print(f"{ff}")

quit()

load_opts = io.LoadOpts(delimiter=",",header_rows=0,)

exp_loaders = [
    io.PointSensLoader(
        load_file=data_path/"file",
        sens_key="TCs-0",
        sens_cols=[],
        load_opts=load_opts,
        time_col=0,
        time_slice=slice(None),                    
    ),
    io.PointSensLoader(
        load_file=data_path/"file",
        sens_key="TCs-1",
        sens_cols=[],
        load_opts=load_opts,
        time_col=0,
        time_slice=slice(None),                    
    ),
    io.PointSensLoader(
        load_file=data_path/"file",
        sens_key="CV",
        sens_cols=[2,],
        load_opts=load_opts,
        time_col=0,
        time_slice=slice(None),                    
    ),
]

exp_data = io.load_exp_data(exp_loaders)

print("Keys in exp_data.fields:")
for kk,aa in exp_data.fields.items():
    print(f"exp_data.fields[{kk}].shape={exp_data.fields[kk].shape}")    
