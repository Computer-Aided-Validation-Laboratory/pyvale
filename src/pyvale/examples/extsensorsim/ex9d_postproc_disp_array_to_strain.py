# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Extended Example 9d: Spatial Differentiation (Displacement Array to Strain)
================================================================================

In experimental biomechanics and full-field structural validation, arrays of
discrete displacement probes (e.g. optical tracking markers, LVDTs, or fiber
gratings) capture displacement vectors :math:`\\mathbf{u}_i(t)` at known
locations :math:`\\mathbf{x}_i`. Reconstructing continuous strain fields
requires 2D/3D spatial surface fitting followed by spatial differentiation:
.. math::
    \\boldsymbol{\\varepsilon}(\\mathbf{x}, t) = \\frac{1}{2}\\left(
        \\nabla \\mathbf{u} + (\\nabla \\mathbf{u})^T
    \\right)

In this example, we demonstrate:
1. Simulating an array of 16 displacement probes across a plate under tension.
2. Fitting spatial displacement polynomials with `ProcessSpatialStrain`.
3. Calculating full infinitesimal strain components :math:`(\\varepsilon_{xx},
   \\varepsilon_{yy}, \\varepsilon_{xy})`.
4. Comparing reconstructed strains against FE simulation truth.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from pyvale.sensorsim.measurementdata import MeasurementData
from pyvale.sensorsim.postprocessspatial import ProcessSpatialStrain


def main(show_plots: bool = False) -> None:
    # --------------------------------------------------------------------------
    # 1. Setup Synthetic Non-Uniform 2D Displacement Field
    # u_x(x,y) = 0.001*x^2 + 0.0005*x*y
    # u_y(x,y) = -0.0003*y^2 - 0.0002*x*y
    # Exact Strains:
    # eps_xx = du_x/dx = 0.002*x + 0.0005*y
    # eps_yy = du_y/dy = -0.0006*y - 0.0002*x
    # eps_xy = 0.5*(du_x/dy + du_y/dx) = 0.5*(0.0005*x - 0.0002*y)

    grid_x, grid_y = np.meshgrid(
        np.linspace(0.0, 50.0, 4), np.linspace(0.0, 50.0, 4)
    )
    x_probes = grid_x.ravel()
    y_probes = grid_y.ravel()
    z_probes = np.zeros_like(x_probes)
    probe_positions = np.column_stack([x_probes, y_probes, z_probes])

    u_x_true = 0.001 * (x_probes**2) + 0.0005 * x_probes * y_probes
    u_y_true = -0.0003 * (y_probes**2) - 0.0002 * x_probes * y_probes

    # Add realistic measurement noise to probe readings
    rng = np.random.default_rng(99)
    u_x_meas = u_x_true + rng.normal(0.0, 0.002, size=u_x_true.shape)
    u_y_meas = u_y_true + rng.normal(0.0, 0.002, size=u_y_true.shape)

    disp_data = MeasurementData(
        values=np.stack([u_x_meas, u_y_meas], axis=1)[:, :, np.newaxis],
        sample_times=np.array([0.0]),
        positions=probe_positions,
        components=("u_x", "u_y"),
        units="mm",
    )

    # --------------------------------------------------------------------------
    # 2. Reconstruct Strain Tensor at Dense Grid Points
    dense_x, dense_y = np.meshgrid(
        np.linspace(5.0, 45.0, 10), np.linspace(5.0, 45.0, 10)
    )
    eval_pts = np.column_stack([
        dense_x.ravel(), dense_y.ravel(), np.zeros(100)
    ])

    strain_processor = ProcessSpatialStrain(
        source="disp_markers",
        poly_degree=2,
        eval_positions=eval_pts,
        spatial_dims="2D",
    )
    strain_res = strain_processor.process({"disp_markers": disp_data})
    calc_strains = strain_res.values[:, :, 0]

    # Exact strains at evaluation points
    x_e = eval_pts[:, 0]
    y_e = eval_pts[:, 1]
    eps_xx_exact = 0.002 * x_e + 0.0005 * y_e
    eps_yy_exact = -0.0006 * y_e - 0.0002 * x_e

    # --------------------------------------------------------------------------
    # 3. Plotting & Verification
    out_dir = Path("pyvale-output/extsensorsim")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel 1: ε_xx comparison
    axes[0].plot(
        x_e, eps_xx_exact, "k.", alpha=0.4, label="Exact FE Truth ε_xx"
    )
    axes[0].plot(
        x_e,
        calc_strains[:, 0],
        "ro",
        markersize=5,
        label="Reconstructed from Displacements",
    )
    axes[0].set_xlabel("Plate Coordinate x (mm)")
    axes[0].set_ylabel("Normal Strain ε_xx")
    axes[0].set_title("Reconstructed Longitudinal Strain ε_xx")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    # Panel 2: ε_yy comparison
    axes[1].plot(
        y_e, eps_yy_exact, "k.", alpha=0.4, label="Exact FE Truth ε_yy"
    )
    axes[1].plot(
        y_e,
        calc_strains[:, 1],
        "bs",
        markersize=5,
        label="Reconstructed from Displacements",
    )
    axes[1].set_xlabel("Plate Coordinate y (mm)")
    axes[1].set_ylabel("Transverse Strain ε_yy")
    axes[1].set_title("Reconstructed Transverse Strain ε_yy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(out_dir / "ext_ex9d_disp_array_to_strain.png", dpi=150)

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main(show_plots=True)
