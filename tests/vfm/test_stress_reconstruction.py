from pathlib import Path

import numpy as np
import pytest
from plots import (
    plot_stress_abs_diff,
    plot_stress_abs_perc_diff,
)
from rms import rms

from pyvale.vfm.constlaws import IsotropicVonMisesElastoplasticity
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.hardening import HardeningLinear

EXPERIMENT_DATA_FILE = (
    Path(__file__).parent
    / "input"
    / "hole2d_plas"
    / "experiment_data.yaml"
)

KNOWN_PARAMETERS_FILE = (
    Path(__file__).parent
    / "gold"
    / "hole2d_plas.npz"
)

KNOWN_STRESS_FILE = (
    Path(__file__).parent
    / "gold"
    / "hole2d_plas_stress.npy"
)

PLOT_STRESS_RECON_ABS_DIFF = False
PLOT_STRESS_RECON_ABS_PERC_DIFF = False


@pytest.mark.skip(reason="known stress file hasn't yet been generated")
def test_stress_reconstruction():
    experiment_data = ExperimentData.load_from_file(EXPERIMENT_DATA_FILE)

    strain = experiment_data.strain
    x = experiment_data.specimen_geometry.x
    y = experiment_data.specimen_geometry.y

    known_parameter_maps = dict(np.load(KNOWN_PARAMETERS_FILE))
    known_stress = np.load(KNOWN_STRESS_FILE)

    # ------------------------------------------------------------------
    # Test the stress reconstruction: reconstruct stress from the known
    # homogeneous parameters and compare against the FE stress.
    # ------------------------------------------------------------------
    print("Reconstructing stress...")

    constitutive_law = IsotropicVonMisesElastoplasticity(HardeningLinear())
    stress_calc = constitutive_law.calculate_stress(strain, known_parameter_maps)

    # abs difference between calculated and known (FE) stress at final timestep
    stress_abs_diff = np.abs(stress_calc[-1] - known_stress[-1])  # shape: (3, y, x)

    if PLOT_STRESS_RECON_ABS_DIFF:
        plot_stress_abs_diff(x, y, stress_abs_diff)

    stress_abs_perc_diff = np.full_like(known_stress[-1], np.nan, dtype=np.float64)
    valid = np.abs(known_stress[-1]) > 0.01 #avoid division by zero
    stress_abs_perc_diff[valid] = (
        np.abs(stress_calc[-1][valid] - known_stress[-1][valid])
        / np.abs(known_stress[-1][valid])
    ) * 100.0  # shape: (3, y, x)

    if PLOT_STRESS_RECON_ABS_PERC_DIFF:
        plot_stress_abs_perc_diff(x, y, stress_abs_perc_diff)

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
