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
    """
    """
    ALL = enum.auto()
    SPLIT = enum.auto()


@dataclass(slots=True)
class ExpSimOpts:
    """
    """
    store_rand_errs: bool = True
    """
    """
    store_sys_errs: bool = True
    """
    """
    workers: int | None = None
    """
    """
    para: EExpSimPara = EExpSimPara.ALL
    """
    """


@dataclass(slots=True)
class ExpSimKeys:
    meas: str = "meas"
    sys: str = "sys_errs"
    rand: str = "rand_errs"
    time: str = "samp_times"


#TODO:
# - probably shouldn't keep a copy of the exp_data - just emit it and the user
#   is responsible for it.
class ExperimentSimulator:
    """An experiment simulator for running monte-carlo simulation by applying a
    dictionary of sensor arrays to a dictionary of simulations over a given
    number of user defined experiments.
    """
    __slots__ = ("_sim_dict","_sens_dict","_num_exp_per_sim","_exp_data",
                 "_exp_sim_opts","_exp_sim_keys")

    def __init__(self,
                 sim_dict: dict[str,mh.SimData],
                 sensor_arrays: dict[str,ISensorArray],
                 exp_sim_opts: ExpSimOpts | None = None,
                 exp_sim_keys: ExpSimKeys | None = None,
                 ) -> None:
        """
        Parameters
        ----------
        sim_dict : dict[str,mh.SimData]
            _description_
        sensor_arrays : dict[str,ISensorArray]
            The sensor arrays that will be applied to each simulation to
            generate the virtual experiment data.

        """

        self._sim_dict = sim_dict
        self._sens_dict = sensor_arrays
        self._num_exp_per_sim = 1
        self._exp_data = None

        if exp_sim_opts is None:
            self._exp_sim_opts = ExpSimOpts()
        else:
            self._exp_sim_opts = exp_sim_opts

        if exp_sim_keys is None:
            self._exp_sim_keys = ExpSimKeys()
        else:
            self._exp_sim_keys = exp_sim_keys


    def get_exp_sim_keys(self) -> ExpSimKeys:

        return self._exp_sim_keys

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
        return self._sens_dict

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
        self._exp_data = {}

        # 1) para over sim_data/sens_array, run N per worker
        if (self._exp_sim_opts.workers is not None and
            self._exp_sim_opts.para == EExpSimPara.ALL):

            assert self._exp_sim_opts.workers > 0, ("Number of workers must"
                        + " be greater than 0.")

            with Pool(self._exp_sim_opts.workers) as pool:
                processes = []
                for (key_sim, sim_data), (key_sens, sens_array) in (
                    product(self._sim_dict.items(), self._sens_dict.items())
                ):
                    time_key = (key_sim,key_sens,self._exp_sim_keys.time)
                    self._exp_data[time_key] = sens_array.get_sample_times()

                    args = (
                        key_sim,
                        key_sens,
                        sim_data,
                        sens_array,
                        self._num_exp_per_sim,
                        self._exp_sim_keys,
                    )

                    process = pool.apply_async(_run_all_sims,args=args)

                    processes.append(process)

                for pp in processes:
                    # dict[tuple[str,str,str],
                    #      shape=(n_exps,n_sens,n_comps,n_time_steps)]
                    sim_exps = pp.get()
                    self._exp_data.update(sim_exps)

        # 2) para over all sim_data/sens_array/Nsims
        elif (self._exp_sim_opts.workers is not None and
              self._exp_sim_opts.para == EExpSimPara.SPLIT):

            assert self._exp_sim_opts.workers > 0, ("Number of workers must"
                                           + " be greater than 0.")

            # We are going to have to populate the experiment array on the fly
            # so we need to pre-alloc to index into it as we get results.
            for (key_sim, sim_data), (key_sens, sens_array) in (
                product(self._sim_dict.items(), self._sens_dict.items())
            ):
                exp_shape = ((self._num_exp_per_sim,)
                    + sens_array.get_measurement_shape())

                self._exp_data[(key_sim,key_sens,"meas")] = np.zeros(
                    exp_shape,dtype=np.float64
                )
                self._exp_data[(key_sim,key_sens,"sys_errs")] = np.zeros(
                    exp_shape,dtype=np.float64
                )
                self._exp_data[(key_sim,key_sens,"rand_errs")] = np.zeros(
                    exp_shape,dtype=np.float64
                )


            with Pool(self._exp_sim_opts.workers) as pool:

                processes_with_id = []
                for (key_sim, sim_data), (key_sens, sens_array) in (
                   product(self._sim_dict.items(), self._sens_dict.items())
                ):
                    time_key = (key_sim,key_sens,self._exp_sim_keys.time)
                    self._exp_data[time_key] = sens_array.get_sample_times()

                    for ee in range(self._num_exp_per_sim):

                        args = (key_sim,
                                key_sens,
                                sim_data,
                                sens_array,
                                self._exp_sim_keys)
                                
                        process = pool.apply_async(_run_one_sim,args=args)

                        processes_with_id.append({"process":process,
                                                  "exp_ind":ee})

                for pp in processes_with_id:
                    # dict[tuple[str,..],shape=(n_sens,n_comps,n_time_steps)]
                    one_exp_dict = pp["process"].get()
                    exp_i = pp["exp_ind"]

                    for kk,aa in one_exp_dict.items():
                        # shape=(n_exps,n_sens,n_comps,n_time_steps)
                        self._exp_data[kk][exp_i,:,:,:] = aa

        else: # 3) Run everything sequentially
            for (key_sim, sim_data), (key_sens, sens_array) in (
                product(self._sim_dict.items(), self._sens_dict.items())
            ):

                sens_array.get_field().set_sim_data(self._sim_dict[key_sim])

                time_key = (key_sim,key_sens,self._exp_sim_keys.time)
                self._exp_data[time_key] = sens_array.get_sample_times()

                exp_res = _run_all_sims(key_sim,
                                        key_sens,
                                        sim_data,
                                        sens_array,
                                        self._num_exp_per_sim,
                                        self._exp_sim_keys)

                self._exp_data.update(exp_res)

                
        # dict[tuple[str,...],shape=(n_sims,n_exps,n_sens,n_comps,n_time_steps)]
        return self._exp_data

    def get_exp_sim_data(self) -> dict[tuple[str,...],np.ndarray] | None:
        """Gets the experiment data dictionary of numpy arrays if simulated
        experiments have been run. Otherwise returns None.

        Returns
        -------
        dict[tuple[str,...],np.ndarray] | None
            Dicitionary of virtual experimental data arrays where the key
            corresponds to the virtual sensor array and the data is an array
            with shape=(n_exps,n_sens,n_comps,n_time_steps). Returns
            None if no virtual experiments have been run.
        """

        # dict[tuple[str,...],shape=(n_exps,n_sens,n_comps,n_time_steps)]
        return self._exp_data


def _run_one_sim(sim_key: str,
                 sens_key: str,
                 sim_data: mh.SimData,
                 sens_array: ISensorArray,
                 exp_keys: ExpSimKeys) -> dict[tuple[str,...],np.ndarray]:

    sens_array.get_field().set_sim_data(sim_data)

    meas = sens_array.sim_measurements()
    sys_errs = sens_array.get_errors_systematic()
    rand_errs = sens_array.get_errors_random()

    sim_exp = {(sim_key,sens_key,exp_keys.meas):meas,
               (sim_key,sens_key,exp_keys.sys):sys_errs,
               (sim_key,sens_key,exp_keys.rand):rand_errs,}

    # RETURN: dict[str,np.array.shape=(n_sens,n_comps,n_time_steps)]
    return sim_exp


def _run_all_sims(sim_key: str,
                  sens_key: str,
                  sim_data: mh.SimData,
                  sens_array: ISensorArray,
                  num_exp: int,
                  exp_keys: ExpSimKeys,
                  ) -> dict[tuple[str,...],np.ndarray]:

    sens_array.get_field().set_sim_data(sim_data)

    exp_shape = (num_exp,)+sens_array.get_measurement_shape()
    meas = np.empty(exp_shape,dtype=np.float64)
    sys_errs = np.empty(exp_shape,dtype=np.float64)
    rand_errs = np.empty(exp_shape,dtype=np.float64)

    for ee in range(num_exp):
        meas[ee,:,:,:] = sens_array.sim_measurements()
        sys_errs[ee,:,:,:] = sens_array.get_errors_systematic()
        rand_errs[ee,:,:,:] = sens_array.get_errors_random()

    sim_exp = {(sim_key,sens_key,exp_keys.meas):meas,
               (sim_key,sens_key,exp_keys.sys):sys_errs,
               (sim_key,sens_key,exp_keys.rand):rand_errs,}

    # RETURN: np.array.shape=(n_exps,n_sens,n_comps,n_time_steps)
    return sim_exp
