# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class ExperimentStats:
    """Dataclass holding summary statistics for a series of simulated
    experiments produced using the experiment simulator. All summary statistics
    are calculated over the 'experiments' dimension of the measurements array so
    the arrays of statistics have the shape=(n_sims,n_sensors,n_field_comps,
    n_time_steps). Note that the n_sims dimension refers to the number of input
    multi-physics simulations (i.e. SimData objects) that the virtual
    experiments were performed over.
    """

    mean: np.ndarray | None = None
    """Mean of each sensors measurement for the given field component and time
    step as an array with shape=(n_sims,n_sensors,n_field_comps,n_time_steps).
    """

    std: np.ndarray | None = None
    """Standard deviation of the sensor measurements for the given field
    component and time step as an array with shape=(n_sims,n_sensors,
    n_field_comps, n_time_steps)
    """

    max: np.ndarray | None = None
    """Maximum of the sensor measurements for the given field component and time
    step as an array with shape=(n_sims,n_sensors,n_field_comps,n_time_steps)
    """

    min: np.ndarray | None = None
    """Minmum of the sensor measurements for the given field component and time
    step as an array with shape=(n_sims,n_sensors,n_field_comps,n_time_steps)
    """

    med: np.ndarray | None = None
    """Median  of the sensor measurements for the given field component and time
    step as an array with shape=(n_sims,n_sensors,n_field_comps,n_time_steps)
    """

    q25: np.ndarray | None = None
    """Lower 25% quantile of the sensor measurements for the given field
    component and time step as an array with shape=(n_sims,n_sensors,
    n_field_comps, n_time_steps)
    """

    q75: np.ndarray | None = None
    """Upper 75% quantile of the sensor measurements for the given field
    component and time step as an array with shape=(n_sims,n_sensors,
    _field_comps, n_time_steps)
    """

    mad: np.ndarray | None = None
    """Median absolute deviation of the sensor measurements for the given field
    component and time step as an array with shape=(n_sims,n_sensors,
    n_field_comps, n_time_steps)
    """


def calc_experiment_stats(exp_data: dict[str,np.ndarray]
                          ) -> dict[str,ExperimentStats]:
    """Calculates summary statistics over all virtual experiments for all 
    virtual sensor arrays. 

    Returns
    -------
    dict[str,ExperimentStats]
        Dicitionary of summary statistics data classes for the virtual 
        experiments. The list index correponds to the virtual sensor array.
    """
    
    # dict[str,shape=(n_sims,n_exps,n_sens,n_comps,n_time_steps)]
    exp_stats: dict[str,ExperimentStats] = {
       kk: ExperimentStats() for kk in exp_data
    }

    for kk,dd in exp_data.items():
        if isinstance(dd,np.ndarray):
            exp_stats[kk] = calc_sensor_array_stats(dd)
        else:
            exp_stats[kk] = dd
                
    # dict[str,shape=(n_sims,n_exps,n_sens,n_comps,n_time_steps)]
    return exp_stats


def calc_sensor_array_stats(exp_data: np.ndarray) -> ExperimentStats:
    """Calculates summary statistics for a specific sensor array over all 
    virual experiments. 

    Returns
    -------
    ExperimentStats
        Summary statistics data classes for the sensor array.
    """
    
    exp_stats = ExperimentStats(
        max = np.max(exp_data,axis=1),
        min = np.min(exp_data,axis=1),
        mean = np.mean(exp_data,axis=1),
        std = np.std(exp_data,axis=1),
        med = np.median(exp_data,axis=1),
        q25 = np.quantile(exp_data,0.25,axis=1),
        q75 = np.quantile(exp_data,0.75,axis=1),
        mad = np.median(np.abs(exp_data -
            np.median(exp_data,axis=1,keepdims=True)),axis=1),
    )
    return exp_stats
