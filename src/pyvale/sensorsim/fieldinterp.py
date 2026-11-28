# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np


def interp_to_sample_time(sample_at_sim_time: np.ndarray,
                          sim_time_steps: np.ndarray,
                          sample_times: np.ndarray,
                          ) -> np.ndarray:
    """Helper function for linear temporal interpolation of simulation data.
    Assumes the input data has already been interpolated to the desired spatial
    locations and then performs temporal interpolation linearly based o the
    specified sample times.

    Parameters
    ----------
    sample_at_sim_time : np.ndarray
        Array of simulated measurement points that have been spatially
        interpolated to the desired location. Has the same shape as standard
        measurement array, shape=(num_sensors,num_field_components,
        num_sim_time_steps)
    sim_time_steps : np.ndarray
        Vector of simulation time steps which should have the same length as the
        last axis of the `sample_at_sim_time` array above.
    sample_times : np.ndarray
        Vector of times at which to sample by linearly interpolating between
        simulation time steps.

    Returns
    -------
    np.ndarray
        Simulated measurement array with shape=(num_sensors,num_field_components
        ,num_sample_times)
    """
    def sample_time_interp(x):
        return np.interp(sample_times, sim_time_steps, x)

    n_time_steps = sample_times.shape[0]
    n_sensors = sample_at_sim_time.shape[0]
    n_comps = sample_at_sim_time.shape[1]
    sample_at_spec_time = np.empty((n_sensors,n_comps,n_time_steps))

    for ii in range(n_comps):
        sample_at_spec_time[:,ii,:] = np.apply_along_axis(sample_time_interp,-1,
                                            sample_at_sim_time[:,ii,:])

    return sample_at_spec_time
