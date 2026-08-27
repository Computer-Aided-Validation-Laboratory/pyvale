# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Extended Example 9a: Temporal Differentiation and Noise Filtering
================================================================================

In experimental vibration and kinematics testing, displacement transducers
record physical displacement :math:`u(t)` corrupted by measurement noise.
To derive velocity :math:`v(t) = \\dot{u}(t)` and acceleration
:math:`a(t) = \\ddot{u}(t)`, pre-smoothing is critical to suppress
high-frequency noise amplification during numerical differentiation.

In this example, we demonstrate:
1. Simulating an LVDT displacement sensor measuring dynamic harmonic motion.
2. Filtering raw noisy signals with `ProcessFilterSavitzkyGolay` and
   `ProcessFilterGaussian`.
3. Numerically differentiating smoothed signals with `ProcessDifferentiateTime`
   to recover velocity and acceleration.
4. Comparing raw differentiation vs filtered differentiation against truth.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from pyvale import verif
from pyvale.sensorsim.sensorlibrary import SensorLibrary
from pyvale.sensorsim.measurementdata import MeasurementData
from pyvale.sensorsim.postprocessfilters import (
    ProcessFilterSavitzkyGolay,
    ProcessFilterGaussian,
)
from pyvale.sensorsim.postprocesstemporal import ProcessDifferentiateTime
from pyvale.sensorsim.enums import EDim


def main(show_plots: bool = False) -> None:
    # --------------------------------------------------------------------------
    # 1. Setup Simulation Case with Dynamic Harmonic Displacement
    sim_data, _ = verif.scalar_quadratic_2d()
    n_pts = sim_data.coords.shape[0]
    times = np.linspace(0.0, 1.0, 200)
    sim_data.time = times
    n_times = len(times)

    omega = 2.0 * np.pi * 3.0  # 3 Hz vibration
    u_exact = 2.0 * np.sin(omega * times)

    sim_data.node_vars["disp_x"] = np.outer(np.ones(n_pts), u_exact)
    sim_data.node_vars["disp_y"] = np.zeros((n_pts, n_times))
    sim_data.node_vars["disp_z"] = np.zeros((n_pts, n_times))

    # --------------------------------------------------------------------------
    # 2. Build LVDT Transducer with Measurement Noise
    lvdt = SensorLibrary.lvdt(
        sim_data,
        target_position=(5.0, 3.75, 0.0),
        axis=(1.0, 0.0, 0.0),
        spatial_dims=EDim.TWOD,
        with_meas_errs=True,
    )

    meas_data = MeasurementData.from_sensor_array(lvdt, use_truth=False)
    truth_data = MeasurementData.from_sensor_array(lvdt, use_truth=True)

    # --------------------------------------------------------------------------
    # 3. Post-Processing Pipeline: Smoothing -> Velocity -> Acceleration
    savgol_filter = ProcessFilterSavitzkyGolay(
        source="raw", window_length=15, polyorder=3
    )
    smooth_data = savgol_filter.process({"raw": meas_data})

    diff_v = ProcessDifferentiateTime(
        source="smooth", order=1, label="velocity", units="mm/s"
    )
    vel_data = diff_v.process({"smooth": smooth_data})

    diff_a = ProcessDifferentiateTime(
        source="vel", order=1, label="accel", units="mm/s²"
    )
    accel_data = diff_a.process({"vel": vel_data})

    # Raw differentiation without pre-filtering (noise amplified)
    raw_v = diff_v.process({"smooth": meas_data})

    # Exact analytic derivatives
    v_exact = 2.0 * omega * np.cos(omega * times)
    a_exact = -2.0 * (omega**2) * np.sin(omega * times)

    # --------------------------------------------------------------------------
    # 4. Plotting & Verification
    out_dir = Path("pyvale-output/extsensorsim")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    # Panel 1: Displacement
    axes[0].plot(times, truth_data.values[0, 0], "k--", label="Ground Truth")
    axes[0].plot(
        times, meas_data.values[0, 0], "r.", alpha=0.5, label="Noisy LVDT"
    )
    axes[0].plot(
        times, smooth_data.values[0, 0], "b-", lw=2, label="SavGol Filtered"
    )
    axes[0].set_ylabel("Displacement (mm)")
    axes[0].set_title("Temporal Post-Processing: Displacement to Acceleration")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    # Panel 2: Velocity
    axes[1].plot(times, v_exact, "k--", label="Exact Velocity")
    axes[1].plot(
        times, raw_v.values[0, 0], "r:", alpha=0.6, label="Raw Diff (Noisy)"
    )
    axes[1].plot(
        times, vel_data.values[0, 0], "g-", lw=2, label="Filtered Velocity"
    )
    axes[1].set_ylabel("Velocity (mm/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    # Panel 3: Acceleration
    axes[2].plot(times, a_exact, "k--", label="Exact Accel")
    axes[2].plot(
        times, accel_data.values[0, 0], "m-", lw=2, label="Filtered Accel"
    )
    axes[2].set_ylabel("Accel (mm/s²)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "ext_ex9a_disp_to_velocity.png", dpi=150)

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main(show_plots=True)
