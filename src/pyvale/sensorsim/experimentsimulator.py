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
    dictionary of sensor arrays to a dictionary of simulations over a given
    number of user defined experiments.
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
            _description_
        sensor_arrays : dict[str,ISensorArray]
            The sensor arrays that will be applied to each simulation to
            generate the virtual experiment data.
        reserved_sim_key : str, optional
            String key used for storing the string keys/tags from the simulation
            dictionary in the returned simulated experiment data dictionary, by 
            default "sim_keys".

        Raises
        ------
        ExpSimError
            The reserved dicitionary key has been used in the simulation 
            dictionary. Change the reserved key of the keys used in the 
            simulation dictionary.
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
        """Gets the dicitionary of simulations to run simulated experiments for.

        Returns
        -------
        dict[str,mh.SimData]
            Dictionary of simulation data objects.
        """
        return self._sim_dict

    def get_sensor_array_dict(self) -> dict[str,ISensorArray]:
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
        input simulation cases and virtual sensor arrays returning the results.

        Parameters
        ----------
        num_exp_per_sim : int
            Number of virtual experiments to perform for each simulation and
            sensor array. Must be a non-zero positive integer.

        Returns
        -------
        dict[str,np.ndarray]
            Dicitionary of virtual experimental data arrays where the key
            corresponds to the virtual sensor array and the data is an array
            with shape=(n_sims,n_exps,n_sens,n_comps,n_time_steps). Also, 
            contains the reserved sim key (default='sim_keys') with values that 
            contains the labels for the first simulation axis. 

        Raises
        ------
        ExpSimError
            The number of virtual experiments to run is not a positive integer.
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

        # Stores the simulation string keys as an array in the dictionary,
        # allows user to identify labels for the sim axis in the results array
        self._exp_data[self._reserved_sim_key] = np.array(
            list(self._sim_dict.keys()),
            dtype='U',
        )

        # dict[str,shape=(n_sims,n_exps,n_sens,n_comps,n_time_steps)]
        return self._exp_data

    def get_exp_sim_data(self) -> dict[str,np.ndarray] | None:
        """Gets the experiment data dictionary of numpy arrays if simulated
        experiments have been run. Otherwise returns None.

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


