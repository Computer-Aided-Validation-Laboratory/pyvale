# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2024 The Computer Aided Validation Team
# ==============================================================================

from pathlib import Path
import numpy as np

import pyvale.dataset as dataset
import pyvale.dataio as io
import pyvale.valid as val

data_path = dataset.valid_data_dir()

load_opts = io.LoadOpts(delimiter=",",header_rows=0,)

exp_loaders = [
    io.PointSensLoader(
        load_files=[data_path/"Pulse253_SteadyDICData.csv",
                    data_path/"Pulse254_SteadyDICData.csv",
                    data_path/"Pulse255_SteadyDICData.csv",],
        sens_array_key="TCs-DIC",
        sens_cols=np.arange(2,11),
        sens_labels="TC",                    
    ),
    io.PointSensLoader(
        load_files=[data_path/"Pulse253_SteadyHIVEData.csv",
                    data_path/"Pulse254_SteadyHIVEData.csv",
                    data_path/"Pulse255_SteadyHIVEData.csv",],
        sens_array_key="TCs-HIVE",
        sens_cols=np.array([2,]),
        sens_labels="TC",                    
    ),
    io.PointSensLoader(
        load_files=[data_path/"Pulse253_SteadyPICOData.csv",
                    data_path/"Pulse254_SteadyPICOData.csv",
                    data_path/"Pulse255_SteadyPICOData.csv",],
        sens_array_key="CV",
        sens_cols=np.array([1,]),
        sens_labels="CV",                    
    ),
]

exp_data = io.load_exp_data(exp_loaders)

print(80*"-")
print("Loaded experimental point sensor data.\n")

print("In exp_data.sens_labels:")
for kk,aa in exp_data.sens_labels.items():   
    print(f"exp_data.sens_labels[{kk}]=\n    {exp_data.sens_labels[kk]}")
print()

print("In exp_data.fields:")
for kk,aa in exp_data.fields.items():
    print(f"exp_data.fields[{kk}].shape={exp_data.fields[kk].shape}")    
print(80*"-")
print()

