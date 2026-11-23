# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
SensorSim export validation
================================================================================
"""

import os
import subprocess
import sys
from itertools import count
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyvale.sensorsim as sens
import pyvale.validation as vl

matplotlib.use("Agg")

# %%
SHOW_PLOTS = False
if SHOW_PLOTS:
    plt.switch_backend("QtAgg")

output_path = Path.cwd() / "pyvale-output" / "valid_ex11_sensorsim_import"
output_path.mkdir(parents=True, exist_ok=True)
fig_nums = count(1)

# %%
producer = (
    Path(__file__).resolve().parents[1]
    / "basicsensorsim"
    / "ex3_experiment_simulator.py"
)

producer_environment = {
    **os.environ,
    "MPLBACKEND": "Agg",
    "PYVISTA_OFF_SCREEN": "true",
    "QT_QPA_PLATFORM": "offscreen",
}

with TemporaryDirectory() as producer_directory:
    print("Running SensorSim ex3 to generate the experiment archive...", flush=True)
    subprocess.run(
        (sys.executable, str(producer)),
        cwd=producer_directory,
        env=producer_environment,
        stdout=subprocess.DEVNULL,
        check=True,
    )
    export_path = Path(producer_directory) / "pyvale-output/ex3_exp_sim_data.npz"
    sensor_export = sens.load_exp_sim_data(export_path)

# %%
export_key = ("sim_nominal", "temp")
sens_coords = sensor_export[(*export_key, "sens_pos")]
sens_times = sensor_export[(*export_key, "sens_times")]
exp_temp_measurements = sensor_export[(*export_key, "meas")]

sim_ref_temp = (
    exp_temp_measurements[0]
    - sensor_export[(*export_key, "rand_errs")][0]
    - sensor_export[(*export_key, "sys_errs")][0]
)

# %%
temp_tag = "temperature"

physical_axes = (vl.EValidAxis.POINT, vl.EValidAxis.COMPONENT, vl.EValidAxis.TIME)

temp_observable = vl.ValidObservable(
    components=(temp_tag,),
    geometry=vl.EValidGeometry.POINT,
    coords=sens_coords,
    times=sens_times,
    phys_units="degC",
    coord_units="mm",
    time_units="s",
)

observables = {temp_tag: temp_observable}

sim_arrays = {temp_tag: vl.ValidArray(sim_ref_temp, physical_axes)}

exp_temp_samples = np.moveaxis(exp_temp_measurements, 0, -1)
exp_axes = (*physical_axes, vl.EValidAxis.ALEATORY)
exp_arrays = {temp_tag: vl.ValidArray(exp_temp_samples, exp_axes)}

valid_case = vl.ValidCase(observables=observables, sim=sim_arrays, exp=exp_arrays)

# %%
mavm = vl.MetricMAVM()
valid_result = vl.ValidEngine([mavm]).run(valid_case)

# %%

ecdf_plot = vl.plot_sim_exp_cdfs(
    valid_case, temp_tag, point_idxs=None, time_idx=len(sens_times) - 1
)
vl.add_metric_overlay(ecdf_plot, mavm)

fig_file = f"val_ex11_fig{next(fig_nums):02d}_temperature_ecdfs.png"
ecdf_plot.figure.savefig(output_path / fig_file, bbox_inches="tight")

figure, _ = vl.plot_metric_time_points(
    valid_result,
    temp_tag,
    vl.EMetricType.MAVM,
    point_idxs=tuple(range(len(sens_coords))),
    columns=2,
)

fig_file = f"val_ex11_fig{next(fig_nums):02d}_temperature_mavm_time.png"
figure.savefig(output_path / fig_file, bbox_inches="tight")

if SHOW_PLOTS:
    plt.show()
else:
    plt.close(ecdf_plot.figure)
    plt.close(figure)
