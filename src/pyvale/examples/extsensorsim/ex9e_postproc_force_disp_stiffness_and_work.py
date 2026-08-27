# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Extended Example 9e: Multi-Sensor Fusion (Force and Displacement to Stiffness)
================================================================================

In structural and materials characterization, multiple physical sensors are
combined to derive material constitutive properties:
1. Dynamic secant stiffness: :math:`k(t) = \\frac{F(t)}{u(t)}`
2. Mechanical strain energy / work:
   :math:`W(t) = \\int_0^t F(\\tau) \\dot{u}(\\tau) \\, \\mathrm{d}\\tau`

In this example, we demonstrate:
1. Simulating paired Load Cell (force) and LVDT (displacement) sensors.
2. Filtering noise with `ProcessFilterSavitzkyGolay`.
3. Calculating dynamic stiffness using `ProcessStiffness`.
4. Integrating mechanical work using `ProcessWork`.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from pyvale.sensorsim.measurementdata import MeasurementData
from pyvale.sensorsim.postprocessfilters import ProcessFilterSavitzkyGolay
from pyvale.sensorsim.postprocessderived import (
    ProcessStiffness,
    ProcessWork,
)


def main(show_plots: bool = False) -> None:
    # --------------------------------------------------------------------------
    # 1. Setup Synthetic Tensile Test with Non-Linear Elastic Spring
    # Force F(u) = k0*u + alpha*u^3
    k0 = 500.0  # N/mm
    alpha = 25.0  # N/mm^3

    times = np.linspace(0.0, 3.0, 300)
    u_true = 0.8 * (times**1.5)  # Progressive displacement loading (mm)
    f_true = k0 * u_true + alpha * (u_true**3)  # Resultant contact force (N)

    # Exact work W = 0.5*k0*u^2 + 0.25*alpha*u^4
    w_exact = 0.5 * k0 * (u_true**2) + 0.25 * alpha * (u_true**4)

    # Add realistic load cell and LVDT transducer errors
    rng = np.random.default_rng(77)
    f_meas = f_true + rng.normal(0.0, 8.0, size=f_true.shape)
    u_meas = u_true + rng.normal(0.0, 0.015, size=u_true.shape)

    force_data = MeasurementData(
        values=f_meas[np.newaxis, np.newaxis, :],
        sample_times=times,
        positions=np.zeros((1, 3)),
        components=("force",),
        units="N",
    )

    disp_data = MeasurementData(
        values=u_meas[np.newaxis, np.newaxis, :],
        sample_times=times,
        positions=np.zeros((1, 3)),
        components=("disp",),
        units="mm",
    )

    # --------------------------------------------------------------------------
    # 2. Pre-Smoothing and Multi-Sensor Fusion
    f_filter = ProcessFilterSavitzkyGolay(
        source="f_raw", window_length=15, polyorder=2
    )
    u_filter = ProcessFilterSavitzkyGolay(
        source="u_raw", window_length=15, polyorder=2
    )

    f_smooth = f_filter.process({"f_raw": force_data})
    u_smooth = u_filter.process({"u_raw": disp_data})

    stiff_calc = ProcessStiffness(force="F", disp="u", units="N/mm")
    stiff_res = stiff_calc.process({"F": f_smooth, "u": u_smooth})

    work_calc = ProcessWork(force="F", disp="u", units="mJ")
    work_res = work_calc.process({"F": f_smooth, "u": u_smooth})

    # --------------------------------------------------------------------------
    # 3. Plotting & Verification
    out_dir = Path("pyvale-output/extsensorsim")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel (0, 0): Force vs Displacement
    axes[0, 0].plot(
        u_true, f_true, "k--", label="Exact Constitutive Curve"
    )
    axes[0, 0].plot(
        u_meas, f_meas, "r.", alpha=0.3, label="Raw Transducer Signals"
    )
    axes[0, 0].plot(
        u_smooth.values[0, 0],
        f_smooth.values[0, 0],
        "b-",
        lw=2,
        label="Smoothed Response",
    )
    axes[0, 0].set_xlabel("Displacement u (mm)")
    axes[0, 0].set_ylabel("Force F (N)")
    axes[0, 0].set_title("Nonlinear Force-Displacement Response")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper left")

    # Panel (0, 1): Dynamic Secant Stiffness k(u)
    k_secant_exact = k0 + alpha * (u_true**2)
    valid_idx = times > 0.05
    axes[0, 1].plot(
        u_true[valid_idx],
        k_secant_exact[valid_idx],
        "k--",
        label="Exact Stiffness k(u)",
    )
    axes[0, 1].plot(
        u_smooth.values[0, 0, valid_idx],
        stiff_res.values[0, 0, valid_idx],
        "g-",
        lw=2,
        label="Derived Stiffness F/u",
    )
    axes[0, 1].set_xlabel("Displacement u (mm)")
    axes[0, 1].set_ylabel("Stiffness (N/mm)")
    axes[0, 1].set_title("Derived Secant Stiffness k(u)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc="upper left")

    # Panel (1, 0): Mechanical Work W(t)
    axes[1, 0].plot(times, w_exact, "k--", label="Exact Strain Energy W(t)")
    axes[1, 0].plot(
        times,
        work_res.values[0, 0],
        "m-",
        lw=2,
        label="Integrated Work ∫ F du",
    )
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Mechanical Work (mJ)")
    axes[1, 0].set_title("Cumulative Strain Energy")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc="upper left")

    # Panel (1, 1): Force Time History
    axes[1, 1].plot(times, f_true, "k--", label="Exact Force")
    axes[1, 1].plot(times, f_meas, "r.", alpha=0.3, label="Noisy Load Cell")
    axes[1, 1].plot(
        times, f_smooth.values[0, 0], "b-", label="Filtered Force"
    )
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Force (N)")
    axes[1, 1].set_title("Load Cell Time History")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "ext_ex9e_stiffness_and_work.png", dpi=150)

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main(show_plots=True)
