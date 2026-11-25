# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Load DIC fields and load-cell data
================================================================================

Read two exports of the same plate-with-hole experiment. The column export
stores coordinates and measurements together at each frame. The matrix export
stores each component as a separate spatial grid at each frame. A DAQ table
provides timestamps and the load-cell trace.
"""

from pathlib import Path

import numpy as np

import pyvale.dataio as io
from pyvale.examples.dataio._synthetic_dic_experiment import (
    write_synthetic_dic_experiment,
)

# %%
# 1. Generate the example files
# -----------------------------
# Sample the packaged finite-element plate-with-hole simulation onto a regular
# grid. Unmeasured locations, including the hole, are exported as NaN. The
# load-cell trace uses the top-edge reaction from the simulation.
output_path = Path.cwd() / "pyvale-output" / "dataio_ex03_dic"
files = write_synthetic_dic_experiment(output_path)

# %%
# 2. Read the DAQ table and align frame timestamps
# -----------------------------------------------
# This experiment records one DAQ row per camera frame. The frame ID links the
# rows to the exports, while the timestamp gives the camera time in seconds.
daq_loader = io.PointSensLoader(
    load_files=files.daq_file,
    sens_cols=(0, 2),
    sens_labels=("frame_id", "force_y"),
    time_col=1,
)
daq_data = daq_loader.load_data()
if daq_data.times is None:
    raise ValueError("The DAQ table must provide timestamps.")
frame_ids = tuple(files.column_frames)
recorded_ids = daq_data.fields[daq_data.sens_label_to_ind["frame_id"]]
time_by_frame = dict(zip(recorded_ids.astype(int), daq_data.times, strict=True))
frame_times = np.array([time_by_frame[frame_id] for frame_id in frame_ids])

# %%
# 3. Load both camera exports
# ---------------------------
# The column loader can also read irregular point layouts. It matches the
# rows by point ID, including the shuffled rows in the second frame. The
# matrix loader flattens each spatial grid in row-major order and retains its
# original shape. Both return fields indexed by point, component and time.
camera_loaders: dict[str, io.IExpLoader] = {
    "columns": io.CameraLoaderColumn(
        frame_files=[files.column_frames[frame_id] for frame_id in frame_ids],
        coord_cols=(1, 2, 3),
        component_cols={
            label: column
            for column, label in enumerate(files.component_labels, start=4)
        },
        point_id_col=0,
        times=frame_times,
    ),
    "matrices": io.CameraLoaderMatrix(
        frame_files=[files.matrix_frames[frame_id] for frame_id in frame_ids],
        coord_files=files.coordinate_files,
        times=frame_times,
    ),
}
camera_data = io.load_exp_data(camera_loaders)

# %%
# 4. Recover a strain map from the matrix export
# ----------------------------------------------
matrix_data = camera_data["matrices"]
strain_index = files.component_labels.index("strain_yy")
grid_shape = matrix_data.grid_shape
if grid_shape is None:
    raise ValueError("The matrix export must provide a spatial grid shape.")
final_strain = matrix_data.fields[:, strain_index, -1].reshape(grid_shape)

force_y = daq_data.fields[daq_data.sens_label_to_ind["force_y"]]
for layout, acquisition in camera_data.items():
    print(f"{layout} DIC fields (points, components, frames):", acquisition.fields.shape)
print("Final strain grid:", final_strain.shape)
print("Final load-cell reading [N]:", force_y[-1])
