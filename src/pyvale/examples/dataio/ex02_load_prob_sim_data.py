# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Load probabilistic simulation data
================================================================================

Read thermocouple and coil-voltage simulation samples from a CSV. Each column
is one channel, and each block of 100 rows contains the aleatory samples for one
epistemic case.
"""

from pathlib import Path

import numpy as np

import pyvale.data as dataset
import pyvale.dataio as io

# %%
# 1. Read the simulation table
# ----------------------------
#
# Use ``load_array`` to read the CSV with its first row as a header. The
# resulting table has one row per simulation sample and one column per
# measurement channel.
sim_csv: Path = dataset.valid_data_dir() / "SamplingResultsOnlyPointSensors.csv"
sensor_tags = tuple(f"TC{ii}" for ii in range(1, 11)) + ("CV",)
n_epistemic = 50
n_aleatory = 100
sim_rows: np.ndarray = io.load_array(file_path=sim_csv, header=0, delimiter=",")

# %%
# 2. Organise the uncertainty axes
# --------------------------------
#
# Transpose the table to put channels first. The file groups 100 consecutive
# aleatory samples per epistemic case, so the reshape gives channel, epistemic
# case, and aleatory sample axes. Use the run ordering of your own data to
# choose the reshape.
sim_samples = sim_rows.T.reshape(len(sensor_tags), n_epistemic, n_aleatory)
print("Sensors:", sensor_tags)
print("Simulation shape:", sim_samples.shape, "(sensor, epistemic, aleatory)")
print("First 5 TC3 samples for epistemic case 0 [degC]:", sim_samples[2, 0, :5])
