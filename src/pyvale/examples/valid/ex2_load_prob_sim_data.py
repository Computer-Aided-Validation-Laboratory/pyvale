# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Example 2: Loading Probabilistic Simulation Data for Point Sensors.

Demonstrates loading probabilistic FE simulation outputs with aleatory and
epistemic sample dimensions into a PointValData container.
"""

from pathlib import Path
import numpy as np

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.valid as val


def main() -> None:
    data_path = dataset.valid_data_dir()
    sim_csv = data_path / "SamplingResultsOnlyPointSensors.csv"

    # Column mapping for thermocouple point sensor simulation results
    sens_keys = {
        "TC1": 0,
        "TC2": 1,
        "TC3": 2,
        "TC4": 3,
        "TC5": 4,
        "TC6": 5,
        "TC7": 6,
        "TC8": 7,
        "TC9": 8,
        "TC10": 9,
    }

    # Load 50 epistemic points with 100 aleatory samples each
    n_epistemic = 50
    n_aleatory = 100

    sim_val_data = val.load_prob_sim_csv(
        csv_path=sim_csv,
        sens_keys=sens_keys,
        n_epistemic=n_epistemic,
        n_aleatory=n_aleatory,
    )

    print(80 * "=")
    print("Probabilistic Simulation Data Summary:")
    print(80 * "=")
    sim_arr = sim_val_data.val_points["sim"]
    print(f"Simulation Tensor Shape: {sim_arr.shape}")
    print(f"  Sensors count: {sim_arr.shape[0]}")
    print(f"  Epistemic realizations: {sim_arr.shape[1]}")
    print(f"  Aleatory samples per realization: {sim_arr.shape[2]}")
    print()

    # Calculate p-box envelopes across the epistemic parameter space
    pboxes = val.calc_limit_cdfs_point(sim_val_data)
    print("Computed Epistemic p-boxes:")
    for lbl, (lower_data, upper_data) in pboxes.items():
        print(
            f"  {lbl}: lower mean = {np.mean(lower_data):.2f}, "
            f"upper mean = {np.mean(upper_data):.2f}"
        )


if __name__ == "__main__":
    main()
