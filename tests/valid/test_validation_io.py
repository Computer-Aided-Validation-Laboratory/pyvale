# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Tests for validation data I/O and PointValData batch calculations."""

from pathlib import Path
import numpy as np
import pytest

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.valid as val


def test_pointsensloader_and_extract_val_data() -> None:
    """Load multi-DAQ experimental files and extract PointValData."""
    data_path = dataset.valid_data_dir()

    load_opts = io.ExpLoadOpts(delimiter=",", header_rows=0)
    exp_loaders = {
        "DIC-DAQ": io.PointSensLoader(
            load_files=[
                data_path / "Pulse253_SteadyDICData.csv",
                data_path / "Pulse254_SteadyDICData.csv",
            ],
            sens_cols=np.arange(2, 11),
            sens_labels=[
                "TC1",
                "TC3",
                "TC4",
                "TC5",
                "TC6",
                "TC7",
                "TC8",
                "TC9",
                "TC10",
            ],
            load_opts=load_opts,
        ),
        "HIVE-DAQ": io.PointSensLoader(
            load_files=data_path / "Pulse253_SteadyHIVEData.csv",
            sens_cols=np.array([2]),
            sens_labels=["TC2"],
            load_opts=load_opts,
        ),
    }

    exp_data = io.load_exp_data(exp_loaders)
    assert "DIC-DAQ" in exp_data
    assert "HIVE-DAQ" in exp_data
    assert exp_data["DIC-DAQ"].fields.shape[0] == 9
    assert exp_data["HIVE-DAQ"].fields.shape[0] == 1

    val_data = val.extract_val_data_by_key(
        exp_data,
        sensor_keys={
            "DIC-DAQ": ["TC1", "TC3"],
            "HIVE-DAQ": ["TC2"],
        },
        steady_slice=slice(0, 50),
    )

    assert val_data.val_points["DIC-DAQ"].shape == (2, 50)
    assert val_data.val_points["HIVE-DAQ"].shape == (1, 50)
    assert ("DIC-DAQ", "TC1") in val_data.val_label_to_ind
    assert ("HIVE-DAQ", "TC2") in val_data.val_label_to_ind


def test_load_prob_sim_csv_and_calc_mavm_point() -> None:
    """Load simulation samples and run batch MAVM against experimental data."""
    data_path = dataset.valid_data_dir()

    sim_csv = data_path / "SamplingResultsOnlyPointSensors.csv"
    if not sim_csv.is_file():
        pytest.skip("SamplingResultsOnlyPointSensors.csv not available")

    # Column mapping for thermocouples in simulation results
    sens_keys = {
        "TC2": 1,
        "TC3": 2,
        "TC5": 4,
    }

    # Load a subset of simulation samples (50 epistemic x 20 aleatory = 1000)
    sim_val_data = val.load_prob_sim_csv(
        csv_path=sim_csv,
        sens_keys=sens_keys,
        n_epistemic=50,
        n_aleatory=20,
    )

    assert sim_val_data.val_points["sim"].shape == (3, 50, 20)

    # Load corresponding experimental data
    load_opts = io.ExpLoadOpts(delimiter=",", header_rows=0)
    exp_loaders = {
        "DIC-DAQ": io.PointSensLoader(
            load_files=data_path / "Pulse253_SteadyDICData.csv",
            sens_cols=np.array([3, 5]),
            sens_labels=["TC3", "TC5"],
            load_opts=load_opts,
        ),
        "HIVE-DAQ": io.PointSensLoader(
            load_files=data_path / "Pulse253_SteadyHIVEData.csv",
            sens_cols=np.array([2]),
            sens_labels=["TC2"],
            load_opts=load_opts,
        ),
    }
    exp_data = io.load_exp_data(exp_loaders)
    exp_val_data = val.extract_val_data_by_key(exp_data)

    mavm_results = val.calc_mavm_point(sim_val_data, exp_val_data, alpha=0.05)

    assert "TC2" in mavm_results
    assert "TC3" in mavm_results
    assert "TC5" in mavm_results
    for lbl, res in mavm_results.items():
        assert res.d_total >= 0.0
