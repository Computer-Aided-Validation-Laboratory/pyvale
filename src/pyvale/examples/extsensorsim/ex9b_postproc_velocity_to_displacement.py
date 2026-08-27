# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Extended Example 9b: Temporal Integration (Velocity to Displacement)
================================================================================

Laser Doppler Vibrometers (LDV) and electromagnetic geophones measure physical
velocity :math:`v(t)`. In structural dynamics, recovering displacement
:math:`u(t)` requires numerical integration:
.. math::
    u(t) = \\int_{t_0}^t v(\\tau) \\, \\mathrm{d}\\tau + u_0

In this example, we demonstrate:
1. Simulating a velocity sensor recording harmonic oscillations with noise.
2. Applying a Gaussian pre-smoothing filter with `ProcessFilterGaussian`.
3. Performing cumulative trapezoidal time integration with
   `ProcessIntegrateTime`.
4. Comparing reconstructed displacement against ground truth simulation data.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from pyvale.sensorsim.measurementdata import MeasurementData
from pyvale.sensorsim.postprocessfilters import ProcessFilterGaussian
from pyvale.sensorsim.postprocesstemporal import ProcessIntegrateTime


def main(show_plots: bool = False) -> None:
    # --------------------------------------------------------------------------
    # 1. Generate Synthetic Velocity Sensor Signal with Noise and Offset
    times = np.linspace(0.0, 2.0, 300)
    omega = 2.0 * np.pi * 2.0  # 2 Hz
    amp = 3.5  # mm
    u0 = 5.0  # initial offset mm

    # Exact displacement and velocity
    u_exact = amp * np.sin(omega * times) + u0
    v_exact = amp * omega * np.cos(omega * times)

    # Add realistic sensor noise and small zero-drift
    rng = np.random.default_rng(123)
    v_noisy = v_exact + rng.normal(0.0, 1.5, size=v_exact.shape)

    vel_data = MeasurementData(
        values=v_noisy[np.newaxis, np.newaxis, :],
        sample_times=times,
        positions=np.array([[0.0, 0.0, 0.0]]),
        components=("velocity",),
        units="mm/s",
    )

    # --------------------------------------------------------------------------
    # 2. Post-Processing Pipeline: Gaussian Pre-Filtering -> Time Integration
    gauss_filter = ProcessFilterGaussian(source="vel_raw", sigma=2.0)
    smooth_vel = gauss_filter.process({"vel_raw": vel_data})

    integrator = ProcessIntegrateTime(
        source="vel_smooth",
        initial_value=u0,
        label="displacement",
        units="mm",
    )
    disp_calc = integrator.process({"vel_smooth": smooth_vel})

    # Direct integration without pre-smoothing
    disp_raw_int = integrator.process({"vel_smooth": vel_data})

    # --------------------------------------------------------------------------
    # 3. Plotting & Verification
    out_dir = Path("pyvale-output/extsensorsim")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Panel 1: Velocity
    axes[0].plot(times, v_exact, "k--", label="Exact Velocity")
    axes[0].plot(
        times, v_noisy, "r.", alpha=0.4, label="Measured LDV Velocity"
    )
    axes[0].plot(
        times,
        smooth_vel.values[0, 0],
        "b-",
        lw=2,
        label="Gaussian Smoothed",
    )
    axes[0].set_ylabel("Velocity (mm/s)")
    axes[0].set_title("Temporal Integration: Velocity to Displacement")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    # Panel 2: Reconstructed Displacement
    axes[1].plot(times, u_exact, "k--", label="Exact Displacement")
    axes[1].plot(
        times,
        disp_raw_int.values[0, 0],
        "r:",
        alpha=0.6,
        label="Raw Integration",
    )
    axes[1].plot(
        times,
        disp_calc.values[0, 0],
        "g-",
        lw=2,
        label="Filtered Integration",
    )
    axes[1].set_ylabel("Displacement (mm)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "ext_ex9b_velocity_to_disp.png", dpi=150)

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main(show_plots=True)
