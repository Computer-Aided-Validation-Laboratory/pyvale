import numpy as np
from utils import rms
from load_sim_data import load_strain, load_stress
from plots import (
    _plot_stress_abs_diff,
    _plot_stress_abs_perc_diff,
)

from pyvale.vfm.constlaws import IsotropicVonMisesElastoplasticity
from pyvale.vfm.hardening import LinearHardening

PLOT_STRESS_RECON_ABS_DIFF = False
PLOT_STRESS_RECON_ABS_PERC_DIFF = False


def test_stress_reconstruction():
    exodus_file_name = "out_hole2d_plas_32f.e"
    grid_divs = 101
    known_parameters = {
        "elastic_modulus": 200_000.0,  # MPa
        "poissons_ratio": 0.3,
        "yield_strength": 200.0,       # MPa
        "hardening_modulus": 1_000.0,  # MPa
    }

    (x_grid, y_grid, strain) = load_strain(exodus_file_name, grid_divs)
    (x_grid, y_grid, stress_fe) = load_stress(exodus_file_name, grid_divs)

    constitutive_law = IsotropicVonMisesElastoplasticity(LinearHardening())

    known_parameter_maps = {
        name: np.full((grid_divs, grid_divs), value)
        for name, value in known_parameters.items()
    }

    # ------------------------------------------------------------------
    # Test the stress reconstruction: reconstruct stress from the known
    # homogeneous parameters and compare against the FE stress.
    # ------------------------------------------------------------------
    print("Reconstructing stress...")
    stress_calc = constitutive_law.calculate_stress(strain, known_parameter_maps)

    # abs difference between calculated and known (FE) stress at final timestep
    stress_abs_diff = np.abs(stress_calc[-1] - stress_fe[-1])  # shape: (3, y, x)

    if PLOT_STRESS_RECON_ABS_DIFF:
        _plot_stress_abs_diff(x_grid, y_grid, stress_abs_diff)

    stress_abs_perc_diff = np.full_like(stress_fe[-1], np.nan, dtype=np.float64)
    valid = np.abs(stress_fe[-1]) > 0.01 #avoid division by zero
    stress_abs_perc_diff[valid] = (
        np.abs(stress_calc[-1][valid] - stress_fe[-1][valid])
        / np.abs(stress_fe[-1][valid])
    ) * 100.0  # shape: (3, y, x)

    if PLOT_STRESS_RECON_ABS_PERC_DIFF:
        _plot_stress_abs_perc_diff(x_grid, y_grid, stress_abs_perc_diff)

    stress_abs_diff_mean = float(np.nanmean(stress_abs_diff))
    stress_abs_diff_max = float(np.nanmax(stress_abs_diff))
    stress_abs_diff_rms = rms(stress_abs_diff)

    print(f"stress recon abs diff mean [MPa] = {stress_abs_diff_mean:.6f}")
    print(f"stress recon abs diff max  [MPa] = {stress_abs_diff_max:.6f}")
    print(f"stress recon abs diff rms  [MPa] = {stress_abs_diff_rms:.6f}")

    # The calculated stress reconstruction should be close to the known FE
    # stress, so the abs difference statistics should be small relative to the
    # stress magnitude (~hundreds of MPa).
    assert stress_abs_diff_mean < 0.5
    assert stress_abs_diff_max < 10.0
    assert stress_abs_diff_rms < 1.0
