# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from abc import ABC, abstractmethod
import numpy as np


class FieldInterp(ABC):
    @abstractmethod
    def interp_field(self,
                    points: np.ndarray,
                    sample_times: np.ndarray | None = None,
                    ) -> np.ndarray:
        pass


def interp_to_sample_time(sample_at_sim_time: np.ndarray,
                          sim_time_steps: np.ndarray,
                          sample_times: np.ndarray,
                          ) -> np.ndarray:

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