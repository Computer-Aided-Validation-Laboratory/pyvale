# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Data structures and batch validation orchestration for sensor data."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats

from pyvale.dataio.expdata import ExpData
from pyvale.valid.metrics import (
    MAVMResult,
    calc_mavm_1d,
    calc_mavm_pbox_1d,
)
from pyvale.valid.strategy import IValMetric


@dataclass(slots=True)
class PointValData:
    """Validation data container for point sensors.

    Holds sensor data arrays formatted for probabilistic validation:
    - Simulation array shape: (n_sensors, n_epistemic, n_aleatory)
    - Experimental array shape: (n_sensors, n_repeats)
    """

    val_points: dict[str, np.ndarray] = field(default_factory=dict)
    """Dictionary mapping array key to measurement data array."""

    epistemic_intervals: dict[str, np.ndarray | None] = field(
        default_factory=dict
    )
    """Optional epistemic parameter interval bounds."""

    val_label_to_ind: dict[tuple[str, str], int] = field(
        default_factory=dict
    )
    """Mapping of (array_key, sensor_label) to row index."""

    ind_to_val_label: dict[tuple[str, int], str] = field(
        default_factory=dict
    )
    """Mapping of (array_key, row_index) to sensor label."""

    coords: dict[str, np.ndarray | None] = field(default_factory=dict)
    """Coordinates array per array key."""

    times: dict[str, np.ndarray | None] = field(default_factory=dict)
    """Time vector per array key."""


def extract_val_data_by_key(
    exp_data: ExpData | dict[str, ExpData],
    sensor_keys: dict[str, list[str] | None] | None = None,
    steady_slice: dict[str, slice | None] | slice | None = None,
) -> PointValData:
    """Extracts sensor traces from experimental data into a PointValData
    container.

    Parameters
    ----------
    exp_data : ExpData | dict[str, ExpData]
        Single ExpData object or dictionary of ExpData objects from different
        DAQ systems.
    sensor_keys : dict[str, list[str] | None] | None, optional
        Specific sensor labels to extract per DAQ key. If None, extracts all
        sensors.
    steady_slice : dict[str, slice | None] | slice | None, optional
        Time slice representing the steady-state period to extract. If None,
        uses all time steps.

    Returns
    -------
    PointValData
        Extracted and formatted point validation data structure.
    """
    if isinstance(exp_data, ExpData):
        exp_dict = {"default": exp_data}
    else:
        exp_dict = exp_data

    val_data = PointValData()

    for daq_key, daq_data in exp_dict.items():
        if sensor_keys is not None and daq_key in sensor_keys:
            target_labels = sensor_keys[daq_key]
        else:
            target_labels = None

        if target_labels is None:
            extracted_labels = [
                daq_data.ind_to_sens_label[i]
                for i in range(len(daq_data.ind_to_sens_label))
            ]
            row_indices = list(range(len(extracted_labels)))
        else:
            extracted_labels = target_labels
            row_indices = [daq_data.sens_label_to_ind[s] for s in target_labels]

        # Determine time slice
        if isinstance(steady_slice, dict):
            t_slice = steady_slice.get(daq_key, slice(None))
        elif isinstance(steady_slice, slice):
            t_slice = steady_slice
        else:
            t_slice = slice(None)

        if t_slice is None:
            t_slice = slice(None)

        raw_fields = daq_data.fields[row_indices, :]
        sliced_fields = raw_fields[:, t_slice]

        val_data.val_points[daq_key] = sliced_fields

        for idx, lbl in enumerate(extracted_labels):
            val_data.val_label_to_ind[(daq_key, lbl)] = idx
            val_data.ind_to_val_label[(daq_key, idx)] = lbl

        if daq_data.coords is not None:
            val_data.coords[daq_key] = daq_data.coords[row_indices, :]
        else:
            val_data.coords[daq_key] = None

        if daq_data.times is not None:
            val_data.times[daq_key] = daq_data.times[t_slice]
        else:
            val_data.times[daq_key] = None

    return val_data


def load_prob_sim_csv(
    csv_path: Path,
    sens_keys: dict[str, int],
    n_epistemic: int = 1,
    n_aleatory: int | None = None,
    delimiter: str = ",",
    skip_header: int = 1,
) -> PointValData:
    """Loads probabilistic simulation CSV samples into a PointValData container.

    Parameters
    ----------
    csv_path : Path
        Path to the simulation results CSV file.
    sens_keys : dict[str, int]
        Mapping of sensor name to column index in the CSV.
    n_epistemic : int, optional
        Number of epistemic parameter realizations (default 1).
    n_aleatory : int | None, optional
        Number of aleatory samples per epistemic point. If None, computed from
        total row count.
    delimiter : str, optional
        Delimiter.
    skip_header : int, optional
        Header rows to skip.

    Returns
    -------
    PointValData
        PointValData holding simulation samples of shape
        (n_sensors, n_epistemic, n_aleatory).
    """
    df = pd.read_csv(csv_path, delimiter=delimiter)
    arr = df.to_numpy()

    total_samples = arr.shape[0]
    if n_aleatory is None:
        n_aleatory = total_samples // n_epistemic

    labels = list(sens_keys.keys())
    n_sensors = len(labels)
    sim_tensor = np.zeros(
        (n_sensors, n_epistemic, n_aleatory),
        dtype=np.float64,
    )

    for idx, (lbl, col_idx) in enumerate(sens_keys.items()):
        raw_col = arr[: n_epistemic * n_aleatory, col_idx]
        sim_tensor[idx, :, :] = raw_col.reshape(n_epistemic, n_aleatory)

    val_data = PointValData()
    val_data.val_points["sim"] = sim_tensor

    for idx, lbl in enumerate(labels):
        val_data.val_label_to_ind[("sim", lbl)] = idx
        val_data.ind_to_val_label[("sim", idx)] = lbl

    return val_data


def calc_limit_cdfs_point(
    val_data: PointValData,
    array_key: str = "sim",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Calculates min and max empirical CDFs (p-box) across epistemic samples.

    Parameters
    ----------
    val_data : PointValData
        Validation data container with shape
        (n_sensors, n_epistemic, n_aleatory).
    array_key : str, optional
        Array key to analyze.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        Dictionary mapping sensor label to (pbox_lower_data, pbox_upper_data)
        where lower corresponds to minimum mean / leftmost CDF and upper to
        rightmost CDF.
    """
    data = val_data.val_points[array_key]
    n_sensors, n_epistemic, _ = data.shape

    results = {}
    for s_idx in range(n_sensors):
        lbl = val_data.ind_to_val_label.get((array_key, s_idx), f"S{s_idx}")
        means = np.mean(data[s_idx, :, :], axis=1)
        min_epis = int(np.argmin(means))
        max_epis = int(np.argmax(means))

        pbox_lower = data[s_idx, min_epis, :]
        pbox_upper = data[s_idx, max_epis, :]
        results[lbl] = (pbox_lower, pbox_upper)

    return results


def calc_mavm_point(
    sim_data: PointValData,
    exp_data: PointValData,
    alpha: float = 0.05,
    sim_key: str = "sim",
) -> dict[str, MAVMResult]:
    """Calculates MAVM between simulation and experimental point sensor
    datasets.

    Handles both single-distribution simulations and epistemic p-box
    simulations automatically.

    Parameters
    ----------
    sim_data : PointValData
        Simulation point data container.
    exp_data : PointValData
        Experimental point data container.
    alpha : float, optional
        Significance level (default 0.05).
    sim_key : str, optional
        Simulation array key.

    Returns
    -------
    dict[str, MAVMResult]
        Dictionary mapping sensor label to MAVMResult.
    """
    sim_arr = sim_data.val_points[sim_key]
    results: dict[str, MAVMResult] = {}

    # Find matching sensor labels across exp DAQs
    for (daq_key, exp_lbl), exp_row in exp_data.val_label_to_ind.items():
        if ("sim", exp_lbl) not in sim_data.val_label_to_ind:
            continue

        sim_row = sim_data.val_label_to_ind[("sim", exp_lbl)]
        exp_samples = exp_data.val_points[daq_key][exp_row, :].ravel()

        if sim_arr.ndim == 3 and sim_arr.shape[1] > 1:
            # Epistemic p-box comparison
            means = np.mean(sim_arr[sim_row, :, :], axis=1)
            min_epis = int(np.argmin(means))
            max_epis = int(np.argmax(means))
            sim_min = sim_arr[sim_row, min_epis, :]
            sim_max = sim_arr[sim_row, max_epis, :]
            res = calc_mavm_pbox_1d(
                sim_min,
                sim_max,
                exp_samples,
                alpha=alpha,
            )
        else:
            sim_samples = sim_arr[sim_row, :].ravel()
            res = calc_mavm_1d(
                sim_samples,
                exp_samples,
                alpha=alpha,
            )

        results[exp_lbl] = res

    return results


def calc_metric_point(
    metric: IValMetric,
    sim_data: PointValData,
    exp_data: PointValData,
    sim_key: str = "sim",
) -> dict[str, Any]:
    """Calculates any IValMetric strategy across matching point sensors.

    Parameters
    ----------
    metric : IValMetric
        Validation metric strategy instance.
    sim_data : PointValData
        Simulation data container.
    exp_data : PointValData
        Experimental data container.
    sim_key : str, optional
        Simulation array key.

    Returns
    -------
    dict[str, Any]
        Dictionary mapping sensor label to validation metric output.
    """
    sim_arr = sim_data.val_points[sim_key]
    results: dict[str, Any] = {}

    for (daq_key, exp_lbl), exp_row in exp_data.val_label_to_ind.items():
        if ("sim", exp_lbl) not in sim_data.val_label_to_ind:
            continue

        sim_row = sim_data.val_label_to_ind[("sim", exp_lbl)]
        sim_samples = sim_arr[sim_row, :].ravel()
        exp_samples = exp_data.val_points[daq_key][exp_row, :].ravel()

        results[exp_lbl] = metric.calc(sim_samples, exp_samples)

    return results
