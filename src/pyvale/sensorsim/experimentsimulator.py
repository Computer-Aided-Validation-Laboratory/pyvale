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
from pyvale.sensorsim.exceptions import ExpSimError

class ExperimentSimulator:
    """An experiment simulator for running monte-carlo analysis by applying a
    list of sensor arrays to a list of simulations over a given number of user
    defined experiments. Calculates summary statistics for each sensor array
    applied to each simulation.
    """
    __slots__ = ("_sim_dict","_sensor_arrays","_num_exp_per_sim","_exp_data",
                 "_exp_stats","_reserved_sim_key")

    def __init__(self,
                 sim_dict: dict[str,mh.SimData],
                 sensor_arrays: dict[str,ISensorArray],
                 reserved_sim_key: str = "sim_keys",
                 ) -> None:
        """
        Parameters
        ----------
        sim_dict : dict[str,mh.SimData]
            List of simulation data objects over which the virtual experiments
            will be performed.
        sensor_arrays : dict[str,ISensorArray]
            The sensor arrays that will be applied to each simulation to
            generate the virtual experiment data.
        """
        self._sim_dict = sim_dict
        self._sensor_arrays = sensor_arrays
        self._num_exp_per_sim = 1
        self._exp_data = None
        self._exp_stats = None
        self._reserved_sim_key = reserved_sim_key
    
        if self._reserved_sim_key in self._sim_dict:
            raise ExpSimError(
                f"Reserved key cannot be {self._reserved_sim_key} in the" 
                + "simulation dictionary."
            )

    def get_sim_dict(self) -> dict[str,mh.SimData]:
        """Gets the list of simulations to run simulated experiments for.

        Returns
        -------
        dict[str,mh.SimData]
            Dictionary of simulation data objects.
        """
        return self._sim_dict

    def get_sensor_arrays(self) -> dict[str,ISensorArray]:
        """Gets the sensor array dictionary for this experiment.

        Returns
        -------
        dict[str,ISensorArray]
            Dicitionary of sensor arrays for the simulated experiment.
        """
        return self._sensor_arrays

    def run_experiments(self, 
                        num_exp_per_sim: int
                        ) -> dict[str,np.ndarray]:
        """Runs the specified number of virtual experiments over the number of
        input simulation cases and virtual sensor arrays.

        Parameters
        ----------
        num_exp_per_sim : int
            Number of virtual experiments to perform for each simulation and
            sensor array.

        Returns
        -------
        dict[str,np.ndarray]
            Dicitionary of virtual experimental data arrays where the key
            corresponds to the virtual sensor array and the data is an array
            with shape=(n_sims,n_exps,n_sens,n_comps,n_time_steps).
            TODO! Reserved key.
        """

        if num_exp_per_sim <= 0:
            raise ExpSimError(
                "Number of experiments per sim must be a positive integer"
            )

        self._num_exp_per_sim = num_exp_per_sim

        n_sims = len(self._sim_dict)
        
        # dict[str,shape=(n_sims,n_exps,n_sens,n_comps,n_time_steps)]
        self._exp_data = {}

        for key_sens,sens_array in self._sensor_arrays.items():

            meas_array = np.zeros((n_sims,self._num_exp_per_sim)+
                                   sens_array.get_measurement_shape())

            for jj,key_sims in enumerate(self._sim_dict):
                sens_array.get_field().set_sim_data(self._sim_dict[key_sims])

                for ee in range(self._num_exp_per_sim):
                    meas_array[jj,ee,:,:,:] = sens_array.sim_measurements()

            self._exp_data[key_sens] = meas_array

        self._exp_data[self._reserved_sim_key] = list(self._sim_dict.keys()) 
    
        # dict[str,shape=(n_sims,n_exps,n_sens,n_comps,n_time_steps)]
        return self._exp_data

    def get_exp_data(self) -> dict[str,np.ndarray] | None:
        """Gets the experiment data dictionary of numpy arrays if simulated 
        experiments have been run. Otherwise returns none.

        Returns
        -------
        dict[str,np.ndarray] | None
            Dicitionary of virtual experimental data arrays where the key
            corresponds to the virtual sensor array and the data is an array
            with shape=(n_sims,n_exps,n_sens,n_comps,n_time_steps). Returns 
            None if no virtual experiments have been run.
        """

        # dict[str,shape=(n_sims,n_exps,n_sens,n_comps,n_time_steps)]
        return self._exp_data
         
    
