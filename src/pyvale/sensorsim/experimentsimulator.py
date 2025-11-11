# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
This module is used for performing Monte-Carlo virtual experiments over a series
of input simulation cases and sensor arrays.
"""

import numpy as np
import pyvale.mooseherder as mh
from pyvale.sensorsim.sensorarray import ISensorArray


class ExperimentSimulator:
    """An experiment simulator for running monte-carlo analysis by applying a
    list of sensor arrays to a list of simulations over a given number of user
    defined experiments. Calculates summary statistics for each sensor array
    applied to each simulation.
    """
    __slots__ = ("_sim_list","_sensor_arrays","_num_exp_per_sim","_exp_data",
                 "_exp_stats")

    def __init__(self,
                 sim_list: list[mh.SimData],
                 sensor_arrays: list[ISensorArray],
                 ) -> None:
        """
        Parameters
        ----------
        sim_list : list[mh.SimData]
            List of simulation data objects over which the virtual experiments
            will be performed.
        sensor_arrays : list[ISensorArray]
            The sensor arrays that will be applied to each simulation to
            generate the virtual experiment data.
        """
        self._sim_list = sim_list
        self._sensor_arrays = sensor_arrays
        self._num_exp_per_sim = 1
        self._exp_data = None
        self._exp_stats = None

    def get_sim_list(self) -> list[mh.SimData]:
        """Gets the list of simulations to run simulated experiments for.

        Returns
        -------
        list[mh.SimData]
            List of simulation data objects.
        """
        return self._sim_list

    def get_sensor_arrays(self) -> list[ISensorArray]:
        """Gets the sensor array list for this experiment.

        Returns
        -------
        list[ISensorArray]
            List of sensor arrays for the simulated experiment.
        """
        return self._sensor_arrays

    def run_experiments(self, num_exp_per_sim: int) -> list[np.ndarray]:
        """Runs the specified number of virtual experiments over the number of
        input simulation cases and virtual sensor arrays.

        Parameters
        ----------
        num_exp_per_sim : int
            Number of virtual experiments to perform for each simulation and
            sensor array.

        Returns
        -------
        list[np.ndarray]
            List of virtual experimental data arrays where the list index
            corresponds to the virtual sensor array and the data is an array
            with shape=(n_sims,n_exps,n_sens,n_comps,n_time_steps).
        """

        self._num_exp_per_sim = num_exp_per_sim

        n_sims = len(self._sim_list)
        # shape=list[n_sens_arrays](n_sims,n_exps,n_sens,n_comps,n_time_steps)
        self._exp_data = [None]*len(self._sensor_arrays)

        for ii,aa in enumerate(self._sensor_arrays):
            meas_array = np.zeros((n_sims,self._num_exp_per_sim)+
                                   aa.get_measurement_shape())

            for jj,ss in enumerate(self._sim_list):
                aa.get_field().set_sim_data(ss)

                for ee in range(self._num_exp_per_sim):
                    meas_array[jj,ee,:,:,:] = aa.sim_measurements()

            self._exp_data[ii] = meas_array

        # shape=list[n_sens_arrays](n_sims,n_exps,n_sens,n_comps,n_time_steps)
        return self._exp_data

