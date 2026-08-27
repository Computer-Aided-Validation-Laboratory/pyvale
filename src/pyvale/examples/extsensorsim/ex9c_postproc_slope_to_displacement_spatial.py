# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Extended Example 9c: Spatial Integration (Surface Slope to Displacement)
================================================================================

In experimental deflectometry and optical slope profilometry, sensors record
local surface slope :math:`\\theta(x) = \\frac{\\partial u}{\\partial x}` along
a beam or plate structure. Reconstructing continuous surface deflection
:math:`u(x)` requires spatial line integration from a known anchor boundary:
.. math::
    u(x) = \\int_0^x \\theta(\\xi) \\, \\mathrm{d}\\xi + u_0

In this example, we demonstrate:
1. Simulating an array of 25 slope sensors along a cantilever beam under
   tip load.
2. Integrating discrete slope measurements with `ProcessIntegrateSpatial`.
3. Comparing reconstructed deflection profile against the analytic
   Euler-Bernoulli cantilever deflection
   :math:`u(x) = \\frac{P}{6EI}(3Lx^2 - x^3)`.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from pyvale.sensorsim.measurementdata import MeasurementData
from pyvale.sensorsim.postprocessspatial import ProcessIntegrateSpatial


def main(show_plots: bool = False) -> None:
    # --------------------------------------------------------------------------
    # 1. Analytic Cantilever Beam Deflection and Slope Profiles
    L_beam = 500.0  # mm
    P_tip = 1000.0  # N
    E_mod = 210000.0  # MPa
    I_xx = 5000.0  # mm^4
    EI = E_mod * I_xx

    x_dense = np.linspace(0.0, L_beam, 300)
    # Exact deflection u(x) and slope theta(x)
    u_dense_exact = (P_tip / (6.0 * EI)) * (
        3.0 * L_beam * (x_dense**2) - (x_dense**3)
    )
    theta_dense_exact = (P_tip / (2.0 * EI)) * (
        2.0 * L_beam * x_dense - (x_dense**2)
    )

    # --------------------------------------------------------------------------
    # 2. Discrete Slope Sensor Array with Measurement Noise
    n_sens = 25
    x_sens = np.linspace(0.0, L_beam, n_sens)
    pos_sens = np.column_stack([
        x_sens, np.zeros_like(x_sens), np.zeros_like(x_sens)
    ])

    theta_exact_sens = (P_tip / (2.0 * EI)) * (
        2.0 * L_beam * x_sens - (x_sens**2)
    )

    rng = np.random.default_rng(42)
    noise_slope = rng.normal(0.0, 0.0002, size=theta_exact_sens.shape)
    theta_noisy_sens = theta_exact_sens + noise_slope

    slope_data = MeasurementData(
        values=theta_noisy_sens[:, np.newaxis, np.newaxis],
        sample_times=np.array([0.0]),
        positions=pos_sens,
        components=("slope",),
        units="rad",
    )

    # --------------------------------------------------------------------------
    # 3. Spatial Post-Processing: Integrate Slope along Beam Axis
    spatial_integrator = ProcessIntegrateSpatial(
        source="slope_meas",
        initial_value=0.0,  # Clamped root boundary condition u(0) = 0
        label="deflection",
        units="mm",
    )
    deflection_res = spatial_integrator.process({"slope_meas": slope_data})
    u_reconstructed = deflection_res.values[:, 0, 0]

    # --------------------------------------------------------------------------
    # 4. Plotting & Verification
    out_dir = Path("pyvale-output/extsensorsim")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Panel 1: Slope Profile
    axes[0].plot(
        x_dense, theta_dense_exact, "k--", label="Exact Beam Slope θ(x)"
    )
    axes[0].plot(
        x_sens,
        theta_noisy_sens,
        "ro",
        markersize=5,
        label="Measured Deflectometer Slope",
    )
    axes[0].set_ylabel("Slope θ (rad)")
    axes[0].set_title("Spatial Integration: Slope to Deflection Profile")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    # Panel 2: Deflection Profile
    axes[1].plot(
        x_dense, u_dense_exact, "k--", label="Exact Cantilever Deflection u(x)"
    )
    axes[1].plot(
        x_sens,
        u_reconstructed,
        "b.-",
        lw=2,
        markersize=6,
        label="Reconstructed Deflection (Spatial Integration)",
    )
    axes[1].set_ylabel("Deflection u (mm)")
    axes[1].set_xlabel("Beam Longitudinal Coordinate x (mm)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "ext_ex9c_slope_to_disp.png", dpi=150)

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main(show_plots=True)
