# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
This module is used for performing Monte-Carlo virtual experiments over a series
of input simulation cases and sensor arrays.
"""
import enum
from dataclasses import dataclass
from itertools import product
from multiprocessing.pool import Pool
import numpy as np
import pyvale.mooseherder as mh
from pyvale.sensorsim.sensorarray import ISensorArray
from pyvale.sensorsim.exceptions import ExpSimError


class EExpSimPara(enum.Enum):
    ALL = enum.auto()
    SPLIT = enum.auto()


@dataclass(slots=True)
class ExpSimOpts:
    reserved_sim_key: str = "sim_keys"
    store_truth: bool = False
    store_rand_errs: bool = False
    store_sys_errs: bool = False
    workers: int | None = None
    para: EExpSimPara = EExpSimPara.ALL


class ExperimentSimulator:
    """An experiment simulator for running monte-carlo analysis by applying a
    dictionary of sensor arrays to a dictionary of simulations over a given
    number of user defined experiments.
    """
    __slots__ = ("_sim_dict","_sensor_arrays","_num_exp_per_sim","_exp_data",
                 "_exp_sim_opts")

    def __init__(self,
                 sim_dict: dict[str,mh.SimData],
                 sensor_arrays: dict[str,ISensorArray],
                 exp_sim_opts: ExpSimOpts | None = None,
                 ) -> None:
        """
        Parameters
        ----------
        sim_dict : dict[str,mh.SimData]
            _description_
        sensor_arrays : dict[str,ISensorArray]
            The sensor arrays that will be applied to each simulation to
            generate the virtual experiment data.
    
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
    
        if exp_sim_opts is None:
            self._exp_sim_opts = ExpSimOpts()
        else:
            self._exp_sim_opts = exp_sim_opts

        if self._exp_sim_opts.reserved_sim_key in self._sim_dict:
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

        # pre-alloc exp data arrays
        # dict[str,shape=(n_sims,n_exps,n_sens,n_comps,n_time_steps)]
        self._exp_data = {
            key_sens: np.empty(
                (n_sims, self._num_exp_per_sim) 
                + sens_array.get_measurement_shape(),
                dtype=np.float64,
            )
            for key_sens, sens_array in self._sensor_arrays.items()
        }

        sim_items_with_inds = enumerate(self._sim_dict.items())
 
        # 1) para over sim_data/sens_array, run N per worker
        if (self._exp_sim_opts.workers is not None and 
            self._exp_sim_opts.para == EExpSimPara.ALL):

            assert self._exp_sim_opts.workers > 0, ("Number of threads must"  
                        + " be greater than 0.")

            with Pool(self._exp_sim_opts.workers) as pool:
                processes_with_id = []
                for (sim_ind,(key_sim, sim_data)), (key_sens, sens_array) in (
                    product(sim_items_with_inds, self._sensor_arrays.items())
                ):
                    args = (sim_data,sens_array,self._num_exp_per_sim)
                    process = pool.apply_async(_run_all_sims,args=args)

                    processes_with_id.append({"process":process,
                                              "sim_key":key_sim,
                                              "sim_ind":sim_ind,  
                                              "sens_key":key_sens})
                for pp in processes_with_id:
                    # shape=(n_exps,n_sens,n_comps,n_time_steps)
                    sim_exps = pp["process"].get()
                    sim_ind = pp["sim_ind"]                    
                    # shape=(n_sims,n_exps,n_sens,n_comps,n_time_steps)
                    self._exp_data[pp["sens_key"]][sim_ind,:,:,:,:] = sim_exps
                                    
        # 2) para over all sim_data/sens_array/Nsims
        elif (self._exp_sim_opts.workers is not None and 
              self._exp_sim_opts.para == EExpSimPara.SPLIT):

            assert self._exp_sim_opts.workers > 0, ("Number of threads must"  
                                           + " be greater than 0.")
                   
            with Pool(self._exp_sim_opts.workers) as pool:
                processes_with_id = []
                for (sim_ind,(key_sim, sim_data)), (key_sens, sens_array) in (
                   product(sim_items_with_inds, self._sensor_arrays.items())
                ):
                    for ee in range(self._num_exp_per_sim):
                 
                        args = (sim_data,sens_array)
                        process = pool.apply_async(_run_one_sim,args=args)

                        processes_with_id.append({"process":process,
                                                 "sim_key":key_sim,
                                                 "sim_ind":sim_ind,
                                                 "sens_key":key_sens,
                                                 "exp_ind":ee})

                for pp in processes_with_id:
                    # shape=(n_exps,n_sens,n_comps,n_time_steps)
                    sim_exps = pp["process"].get()                    
                    sim_i = pp["sim_ind"]
                    exp_i = pp["exp_ind"]
                    # shape=(n_sims,n_exps,n_sens,n_comps,n_time_steps)
                    self._exp_data[pp["sens_key"]][sim_i,exp_i,:,:,:] = sim_exps
                                                          
        else: # 3) Run everything sequentially
            for key_sens,sens_array in self._sensor_arrays.items():

                meas_array = np.zeros((n_sims,self._num_exp_per_sim)+
                                       sens_array.get_measurement_shape())

                for jj,key_sims in enumerate(self._sim_dict):
                    sens_array.get_field().set_sim_data(self._sim_dict[key_sims])

                    for ee in range(self._num_exp_per_sim):
                        self._exp_data[key_sens][jj,ee,:,:,:] = (
                            sens_array.sim_measurements()
                        )

        # Stores the simulation string keys as an array in the dictionary,
        # allows user to identify labels for the sim axis in the results array
        self._exp_data[self._exp_sim_opts.reserved_sim_key] = np.array(
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


def _run_one_sim(sim_data: mh.SimData, sens_array: ISensorArray) -> np.ndarray:
    sens_array.get_field().set_sim_data(sim_data)
    # RETURN: np.array.shape=(n_sens,n_comps,n_time_steps)
    return sens_array.sim_measurements()

def _run_all_sims(sim_data: mh.SimData, 
                  sens_array: ISensorArray,
                  num_exp: int,
                  ) -> np.ndarray:

    sens_array.get_field().set_sim_data(sim_data)
    
    sim_experiments = np.zeros((num_exp,)+sens_array.get_measurement_shape())

    for ee in range(num_exp):
        sim_experiments[ee,:,:,:] = sens_array.sim_measurements() 

    # RETURN: np.array.shape=(n_exps,n_sens,n_comps,n_time_steps)
    return sim_experiments
