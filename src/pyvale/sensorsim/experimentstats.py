# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class ExpSimStats:
    """Dataclass holding summary statistics for a series of simulated
    experiments produced using the experiment simulator. All summary statistics
    are calculated over the 'experiments' dimension of the measurements array so
    the arrays of statistics have the shape=(n_sensors,n_field_comps,
    n_time_steps).
    """

    mean: np.ndarray | None = None
    """Mean of each sensors measurement for the given field component and time
    step as an array with shape=(n_sensors,n_field_comps,n_time_steps).
    """

    std: np.ndarray | None = None
    """Standard deviation of the sensor measurements for the given field
    component and time step as an array with shape=(n_sensors,n_field_comps,
    n_time_steps)
    """

    max: np.ndarray | None = None
    """Maximum of the sensor measurements for the given field component and time
    step as an array with shape=(n_sensors,n_field_comps,n_time_steps)
    """

    min: np.ndarray | None = None
    """Minmum of the sensor measurements for the given field component and time
    step as an array with shape=(n_sensors,n_field_comps,n_time_steps)
    """

    med: np.ndarray | None = None
    """Median  of the sensor measurements for the given field component and time
    step as an array with shape=(n_sensors,n_field_comps,n_time_steps)
    """

    q25: np.ndarray | None = None
    """Lower 25% quantile of the sensor measurements for the given field
    component and time step as an array with shape=(n_sensors,n_field_comps,
    n_time_steps)
    """

    q75: np.ndarray | None = None
    """Upper 75% quantile of the sensor measurements for the given field
    component and time step as an array with shape=(n_sensors,n_field_comps,
    n_time_steps)
    """

    mad: np.ndarray | None = None
    """Median absolute deviation of the sensor measurements for the given field
    component and time step as an array with shape=(n_sensors,n_field_comps,
    n_time_steps)
    """


def calc_exp_sim_stats(exp_data: dict[tuple[str,...],np.ndarray]
                          ) -> dict[tuple[str,...],ExpSimStats]:
    """Calculates summary statistics over all virtual experiments for all
    virtual sensor arrays.

    Returns
    -------
    dict[tuple[str,...],ExperimentStats]
        Dictionary of summary statistics data classes for the virtual
        experiments. 
    """

    # dict[tuple[str,..],shape=(n_exps,n_sens,n_comps,n_time_steps)]
    exp_stats: dict[tuple[str,...],ExpSimStats] = {}

    for kk,dd in exp_data.items():
        # f = float
        if dd.dtype.kind == 'f':
            exp_stats[kk] = calc_sensor_array_stats(dd)
        # U = unicode string, S = byte string
        elif dd.dtype.kind == 'U' or dd.dtype.kind == 'S':
            exp_stats[kk] = dd

        # Implicitly remove anything that we don't expect: integers, objects etc

    # dict[tuple[str,..],shape=(n_exps,n_sens,n_comps,n_time_steps)]
    return exp_stats


def calc_sensor_array_stats(exp_data: np.ndarray) -> ExpSimStats:
    """Calculates summary statistics for a specific sensor array over all
    virual experiments.

    Returns
    -------
    ExperimentStats
        Summary statistics data class for the sensor array.
    """

    axis: int = 0
    exp_stats = ExpSimStats(
        max = np.max(exp_data,axis=axis),
        min = np.min(exp_data,axis=axis),
        mean = np.mean(exp_data,axis=axis),
        std = np.std(exp_data,axis=axis),
        med = np.median(exp_data,axis=axis),
        q25 = np.quantile(exp_data,0.25,axis=axis),
        q75 = np.quantile(exp_data,0.75,axis=axis),
        mad = np.median(np.abs(exp_data -
            np.median(exp_data,axis=axis,keepdims=True)),axis=axis),
    )
    return exp_stats
