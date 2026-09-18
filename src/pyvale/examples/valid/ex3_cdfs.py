# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2024 The Computer Aided Validation Team
# ==============================================================================

from pathlib import Path
import numpy as np

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.valid as val

data_path = dataset.valid_data_dir()

load_opts = io.LoadOpts(delimiter=",",header_rows=0,)

exp_loaders = {
    "DIC-DAQ" : io.PointSensLoader(
        load_files=[data_path/"Pulse253_SteadyDICData.csv",
                    data_path/"Pulse254_SteadyDICData.csv",
                    data_path/"Pulse255_SteadyDICData.csv",],
        sens_cols=np.arange(2,11),
        sens_labels=["TC1","TC3","TC4","TC5","TC6","TC7","TC8","TC9","TC10"],
    ),
    "HIVE-DAQ" : io.PointSensLoader(
        load_files=[data_path/"Pulse253_SteadyHIVEData.csv",
                    data_path/"Pulse254_SteadyHIVEData.csv",
                    data_path/"Pulse255_SteadyHIVEData.csv",],
        sens_cols=np.array([2,]),
        sens_labels=["TC2",],                    
    ),
    "PICO-DAQ" : io.PointSensLoader(
        load_files=[data_path/"Pulse253_SteadyPICOData.csv",
                    data_path/"Pulse254_SteadyPICOData.csv",
                    data_path/"Pulse255_SteadyPICOData.csv",],
        sens_cols=np.array([1,]),
        sens_labels=["CV",],
    ),
}

exp_data = io.load_exp_data(exp_loaders)

for kk,ee in exp_data.items():
    print(f"exp_data[{kk}]:")
    print(f"    .field.shape={ee.fields.shape}")
    
    if ee.coords is not None:
        print(f"    .coords.shape={ee.coords.shape}")
    else:
        print(f"    .coords={ee.coords}")

    if ee.times is not None:
        print(f"    .times.shape={ee.times.shape}")
    else:
        print(f"    .times={ee.times}")

    print(f"    .sens_label_to_ind={ee.sens_label_to_ind}")
    print(f"    .ind_to_sens_label={ee.ind_to_sens_label}")

# TODO:
# - Get the experimental data and the simulation data into ValData structures
