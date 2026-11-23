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
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np

import pyvale.sensorsim as sens
import pyvale.validation as vl


def run_sensorsim_import_validation() -> None:
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

    sim_truth_by_repeat = sensor_export.get((*export_key, "truth"))
    if sim_truth_by_repeat is None:
        sim_truth_by_repeat = (
            exp_temp_measurements
            - sensor_export[(*export_key, "rand_errs")]
            - sensor_export[(*export_key, "sys_errs")]
        )
    sim_ref_temp = sim_truth_by_repeat[0]
    np.testing.assert_allclose(
        sim_truth_by_repeat,
        np.broadcast_to(sim_ref_temp, sim_truth_by_repeat.shape),
        rtol=0.0,
        atol=1e-10,
    )

    # %%
    temp_tag = "temperature"
    physical_axes = (vl.EValidAxis.POINT, vl.EValidAxis.COMPONENT, vl.EValidAxis.TIME)
    valid_case = vl.ValidCase(
        {
            temp_tag: vl.ValidObservable(
                components=(temp_tag,),
                geometry=vl.EValidGeometry.POINT,
                coords=sens_coords,
                times=sens_times,
                phys_units="degC",
                coord_units="mm",
                time_units="s",
            )
        },
        {temp_tag: vl.ValidArray(sim_ref_temp, physical_axes)},
        {
            temp_tag: vl.ValidArray(
                np.moveaxis(exp_temp_measurements, 0, -1),
                (*physical_axes, vl.EValidAxis.ALEATORY),
            )
        },
    )

    # %%
    mavm = vl.MetricMAVM()
    valid_result = vl.ValidEngine([mavm]).run(valid_case)
    output_path = Path.cwd() / "pyvale-output" / "valid_ex11_sensorsim_import"
    output_path.mkdir(parents=True, exist_ok=True)

    ecdf_plot = vl.plot_sim_exp_cdfs(
        valid_case, temp_tag, point_idxs=None, time_idx=len(sens_times) - 1
    )
    vl.add_metric_overlay(ecdf_plot, mavm)
    ecdf_plot.figure.savefig(output_path / "temperature_ecdfs.png", bbox_inches="tight")
    plt.close(ecdf_plot.figure)

    figure, _ = vl.plot_metric_time_points(
        valid_result,
        temp_tag,
        vl.EMetricType.MAVM,
        point_idxs=tuple(range(len(sens_coords))),
        columns=2,
    )
    figure.savefig(output_path / "temperature_mavm_time.png", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    run_sensorsim_import_validation()
