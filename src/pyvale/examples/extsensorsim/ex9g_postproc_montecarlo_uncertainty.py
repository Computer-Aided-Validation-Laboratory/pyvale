# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Extended Example 9g: Monte Carlo Uncertainty Propagation in Post-Processing
================================================================================

In virtual test validation, Monte Carlo simulation enables experimentalists to
characterize how physical transducer noise propagates through downstream
analysis pipelines (such as smoothing and numerical differentiation).

In this example, we demonstrate:
1. Orchestrating a 30-trial Monte Carlo virtual experiment in
   `ExperimentSimulator`.
2. Integrating post-processing DAGs directly into the Monte Carlo execution.
3. Quantifying 95% confidence intervals on both raw measurements and derived
   velocity using `calc_exp_sim_stats`.
4. Analyzing the fundamental trade-off between numerical noise amplification
   and smoothing bias.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from pyvale import verif
from pyvale.sensorsim.sensorlibrary import SensorLibrary
from pyvale.sensorsim.experimentsimulator import (
    ExperimentSimulator,
    ExpSimOpts,
    ExpSimSaveKeys,
)
from pyvale.sensorsim.experimentstats import calc_exp_sim_stats
from pyvale.sensorsim.postprocessfilters import ProcessFilterSavitzkyGolay
from pyvale.sensorsim.postprocesstemporal import ProcessDifferentiateTime
from pyvale.sensorsim.enums import EDim


def main(show_plots: bool = False) -> None:
    # --------------------------------------------------------------------------
    # 1. Setup Simulation Case with Dynamic Loading
    sim_data, _ = verif.scalar_quadratic_2d()
    n_pts = sim_data.coords.shape[0]
    times = np.linspace(0.0, 1.0, 150)
    sim_data.time = times
    n_times = len(times)

    omega = 2.0 * np.pi * 2.5
    u_true_t = 1.5 * np.sin(omega * times)
    v_true_t = 1.5 * omega * np.cos(omega * times)

    sim_data.node_vars["disp_x"] = np.outer(np.ones(n_pts), u_true_t)
    sim_data.node_vars["disp_y"] = np.zeros((n_pts, n_times))
    sim_data.node_vars["disp_z"] = np.zeros((n_pts, n_times))

    # --------------------------------------------------------------------------
    # 2. Transducer with Measurement Noise
    lvdt = SensorLibrary.lvdt(
        sim_data,
        target_position=(5.0, 3.75, 0.0),
        axis=(1.0, 0.0, 0.0),
        spatial_dims=EDim.TWOD,
        with_meas_errs=True,
    )

    # --------------------------------------------------------------------------
    # 3. Post-Processing DAG in ExperimentSimulator
    post_procs = {
        "disp_smooth": ProcessFilterSavitzkyGolay(
            source="lvdt", window_length=11, polyorder=2
        ),
        "velocity_derived": ProcessDifferentiateTime(
            source="disp_smooth", order=1, label="velocity", units="mm/s"
        ),
    }

    exp_sim = ExperimentSimulator(
        sim_dict={"dynamic_case": sim_data},
        sensor_arrays={"lvdt": lvdt},
        post_processors=post_procs,
        exp_sim_opts=ExpSimOpts(workers=None),
        exp_save_keys=ExpSimSaveKeys(meas="meas", sens_times="sens_times"),
    )

    n_trials = 30
    exp_data = exp_sim.run_experiments(num_exp_per_sim=n_trials)
    stats = calc_exp_sim_stats(exp_data)

    # Extract Monte Carlo arrays and summary statistics
    u_mc = exp_data[("dynamic_case", "lvdt", "meas")]
    v_mc = exp_data[("dynamic_case", "velocity_derived", "meas")]

    u_mean = stats[("dynamic_case", "lvdt", "meas")].mean[0, 0]
    u_std = stats[("dynamic_case", "lvdt", "meas")].std[0, 0]

    v_mean = stats[("dynamic_case", "velocity_derived", "meas")].mean[0, 0]
    v_std = stats[("dynamic_case", "velocity_derived", "meas")].std[0, 0]

    # --------------------------------------------------------------------------
    # 4. Plotting & Uncertainty Visualization
    out_dir = Path("pyvale-output/extsensorsim")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True)

    # Panel 1: Displacement Monte Carlo Trials and 95% Confidence Band
    for tr in range(min(15, n_trials)):
        axes[0].plot(times, u_mc[tr, 0, 0], "r-", alpha=0.15)
    axes[0].plot(times, u_true_t, "k--", lw=2, label="FE Ground Truth")
    axes[0].plot(times, u_mean, "b-", lw=2, label="Monte Carlo Mean")
    axes[0].fill_between(
        times,
        u_mean - 1.96 * u_std,
        u_mean + 1.96 * u_std,
        color="blue",
        alpha=0.2,
        label="95% Confidence Band (±1.96σ)",
    )
    axes[0].set_ylabel("Displacement (mm)")
    axes[0].set_title(
        f"Monte Carlo Uncertainty ({n_trials} Trials): LVDT Displacement"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    # Panel 2: Derived Velocity Monte Carlo Trials and 95% Confidence Band
    for tr in range(min(15, n_trials)):
        axes[1].plot(times, v_mc[tr, 0, 0], "g-", alpha=0.15)
    axes[1].plot(times, v_true_t, "k--", lw=2, label="Analytic True Velocity")
    axes[1].plot(times, v_mean, "g-", lw=2, label="Derived Velocity Mean")
    axes[1].fill_between(
        times,
        v_mean - 1.96 * v_std,
        v_mean + 1.96 * v_std,
        color="green",
        alpha=0.2,
        label="95% Velocity Uncertainty Band",
    )
    axes[1].set_ylabel("Velocity (mm/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Propagated Uncertainty in Derived Velocity")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "ext_ex9g_postproc_montecarlo.png", dpi=150)

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main(show_plots=True)
