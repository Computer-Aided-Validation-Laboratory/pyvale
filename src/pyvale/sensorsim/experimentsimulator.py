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
    """Parallelisation enumeration for simulated experiments.
    """

    ALL = enum.auto()
    """Each worker performs 'ALL' N simulated experiments for each combination of
    simulation data and sensor arrays. This is the best option of point sensors.
    """

    SPLIT = enum.auto()
    """Simulations are 'SPLIT' across workers each performing 1 of N simulations
    for any given combination of simulation data and sensor arrays. This is the
    best option for when each simulation is computationally heavy such as
    imaging workflows.
    """

@dataclass(slots=True)
class ExpSimKeys:
    """Default string keys for use in the simulated experiment data dictionary.
    these keys appear in the last position of the tuple key and indicate the
    data that is available. For example: simulated measurements default to the
    "meas" key.
    """
    meas: str = "meas"
    sys: str = "sys_errs"
    rand: str = "rand_errs"
    time: str = "samp_times"


@dataclass(slots=True)
class ExpSimOpts:
    """Experiment simulation options dataclass specifying options for what data
    arrays to store in the data dictionary and options for parallelisation of
    the simulated experiments.
    """

    workers: int | None = None
    """Number of workers when running simulations in parallel. Defaults to None.
    If None then simulations are run sequentially without multi-processing.
    """

    para: EExpSimPara = EExpSimPara.ALL
    """Options for running 'ALL' N simulations per worker or 'SPLIT' N
    simulations across workers. 'ALL' is most efficient for point sensors and
    'SPLIT' should be used for computationally heavy single simulations.
    """

    save_errs: bool = True
    """Option to save the systematic and random error arrays to the experiment
    data dictionary.
    """

    exp_sim_keys: ExpSimKeys | None = None
    """Strings keys used in the last position of the tuple key for the output
    simulated experiment data dictionary. If None then the default keys will
    be created and used.
    """

    def __post_init__(self) -> None:
        self.exp_sim_keys = ExpSimKeys()


#TODO:
# - probably shouldn't keep a copy of the exp_data - just emit it and the user
#   is responsible for it.
class ExperimentSimulator:
    """An experiment simulator for running monte-carlo simulation by applying a
    dictionary of sensor arrays to a dictionary of simulations over a given
    number of user defined experiments.
    """
    __slots__ = ("_sim_dict","_sens_dict","_num_exp_per_sim","_exp_data",
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

        """

        self._sim_dict = sim_dict
        self._sens_dict = sensor_arrays
        self._num_exp_per_sim = 1
        self._exp_data = None

        if exp_sim_opts is None:
            self._exp_sim_opts = ExpSimOpts()
        else:
            self._exp_sim_opts = exp_sim_opts


    def get_exp_sim_keys(self) -> ExpSimKeys:
        """Gets the experiment simulation data keys.

        Returns
        -------
        ExpSimKeys
            Dataclass containing the keys used to identify output data from the
            simulated experiments.
        """
        return self._exp_sim_opts.exp_sim_keys

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
            Number of virtual experiments to perform for each combination of
            input physics simulations and sensor arrays. Must be a non-zero
            positive integer.

        Returns
        -------
        dict[tuple[str,...],np.ndarray]
            Dictionary of virtual experimental data arrays where the key is a
            tuple with form (sim_key,sens_key,data_key). The simulation and
            sensor keys correspond to the input simulation and sensor
            dictionaries and the data key returns a given output from the
            simulation. See the `ExpSimKeys` dataclass for valid data keys. The
            data arrays returned for simulated experiment output have shape =
            (n_exps,n_sens,n_comps,n_time_steps).

        Raises
        ------
        ExpSimError
            The number of virtual experiments to run is not a positive integer.
        """

        if num_exp_per_sim <= 0:
            raise ExpSimError(
                "Number of experiments per sim must be a positive integer"
            )

        exp_sim_keys = self._exp_sim_opts.exp_sim_keys
        time_str_key = self._exp_sim_opts.exp_sim_keys.time
        self._num_exp_per_sim = num_exp_per_sim
        self._exp_data = {}

        # 1) para over sim_data/sens_array, run N per worker
        if (self._exp_sim_opts.workers is not None and
            self._exp_sim_opts.para == EExpSimPara.ALL):

            assert self._exp_sim_opts.workers > 0, ("Number of workers must"
                        + " be greater than 0.")

            # Function pointer to avoid if/else inside the hot loop
            if not self._exp_sim_opts.save_errs:
                sim_func = _run_all_sims_meas_only
            else:
                sim_func = _run_all_sims_save_errs

            with Pool(self._exp_sim_opts.workers) as pool:
                processes = []
                for (key_sim, sim_data), (key_sens, sens_array) in (
                    product(self._sim_dict.items(), self._sens_dict.items())
                ):
                    time_key = (key_sim,key_sens,time_str_key)
                    self._exp_data[time_key] = sens_array.get_sample_times()

                    args = (
                        key_sim,
                        key_sens,
                        sim_data,
                        sens_array,
                        self._num_exp_per_sim,
                        exp_sim_keys,
                    )

                    process = pool.apply_async(sim_func,args=args)

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

            # Function pointer to avoid if/else inside the hot loop
            if not self._exp_sim_opts.save_errs:
                sim_func = _run_one_sim_meas_only
            else:
                sim_func = _run_one_sim_save_errs

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
                
                if self._exp_sim_opts.save_errs:
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
                    time_key = (key_sim,key_sens,time_str_key)
                    self._exp_data[time_key] = sens_array.get_sample_times()

                    for ee in range(self._num_exp_per_sim):

                        args = (key_sim,
                                key_sens,
                                sim_data,
                                sens_array,
                                exp_sim_keys)

                        process = pool.apply_async(sim_func,args=args)

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
            # Function pointer to avoid if/else inside the hot loop
            if not self._exp_sim_opts.save_errs:
                sim_func = _run_all_sims_meas_only
            else:
                sim_func = _run_all_sims_save_errs

            for (key_sim, sim_data), (key_sens, sens_array) in (
                product(self._sim_dict.items(), self._sens_dict.items())
            ):

                sens_array.get_field().set_sim_data(self._sim_dict[key_sim])

                time_key = (key_sim,key_sens,time_str_key)
                self._exp_data[time_key] = sens_array.get_sample_times()

                exp_res = sim_func(key_sim,
                                   key_sens,
                                   sim_data,
                                   sens_array,
                                   self._num_exp_per_sim,
                                   exp_sim_keys)

                self._exp_data.update(exp_res)


        # dict[tuple[str,...],shape=(n_sims,n_exps,n_sens,n_comps,n_time_steps)]
        return self._exp_data

    def get_exp_sim_data(self) -> dict[tuple[str,...],np.ndarray] | None:
        """Gets the experiment data dictionary of numpy arrays if simulated
        experiments have been run. Otherwise returns None.

        Returns
        -------
        dict[tuple[str,...],np.ndarray] | None
            Dicitionary of virtual experimental data where the key is tuple of
            the form (sim_key,sens_key,data_key). The simulation and sensor keys
            are the same as input to the simulator. The data key identifys the
            type of output from the virtual experiment. The output simulation
            data are arrays with shape=(n_exps,n_sens,n_comps,n_time_steps).
            Returns None if no virtual experiments have been run.
        """

        # dict[tuple[str,...],shape=(n_exps,n_sens,n_comps,n_time_steps)]
        return self._exp_data


def _run_one_sim_save_errs(sim_key: str,
                           sens_key: str,
                           sim_data: mh.SimData,
                           sens_array: ISensorArray,
                           exp_keys: ExpSimKeys
                           ) -> dict[tuple[str,...],np.ndarray]:
    """Parallelisation helper function for running a single virtual experiment 
    on a worker. This version saves the systematic and random error arrays as
    well as the measurement array.

    Parameters
    ----------
    sim_key : str
        String key identifying the input physics simulation.
    sens_key : str
        String key identifying the sensor array.
    sim_data : mh.SimData
        Simulation data object for the simulation.
    sens_array : ISensorArray
        Sensor array object to apply to the simulation.
    exp_keys : ExpSimKeys
        Contains keys for identifying simulated experiment outputs.
                
    Returns
    -------
    dict[tuple[str,...],np.ndarray]
        Virtual experimental data dictionary.
    """
    
    # NOTE: need to reseed the error chain otherwise each worker inherits the
    # same random seed producing the same simulations.
    sens_array.get_error_integrator().reseed_error_chain()
    sens_array.get_field().set_sim_data(sim_data)

    meas = sens_array.sim_measurements()
    sys_errs = sens_array.get_errors_systematic() 
    rand_errs = sens_array.get_errors_random() 
    
    sim_exp = {(sim_key,sens_key,exp_keys.meas):meas,
               (sim_key,sens_key,exp_keys.sys):sys_errs,
               (sim_key,sens_key,exp_keys.rand):rand_errs}
               
    # RETURN: dict[str,np.array.shape=(n_sens,n_comps,n_time_steps)]
    return sim_exp

def _run_one_sim_meas_only(sim_key: str,
                           sens_key: str,
                           sim_data: mh.SimData,
                           sens_array: ISensorArray,
                           exp_keys: ExpSimKeys
                           ) -> dict[tuple[str,...],np.ndarray]:
    """Parallelisation helper function for running a single virtual experiment 
    on a worker. This version returns only the measurement array.

    Parameters
    ----------
    sim_key : str
        String key identifying the input physics simulation.
    sens_key : str
        String key identifying the sensor array.
    sim_data : mh.SimData
        Simulation data object for the simulation.
    sens_array : ISensorArray
        Sensor array object to apply to the simulation.
    exp_keys : ExpSimKeys
        Contains keys for identifying simulated experiment outputs.
                
    Returns
    -------
    dict[tuple[str,...],np.ndarray]
        Virtual experimental data dictionary.
    """    
    # NOTE: need to reseed the error chain otherwise each worker inherits the
    # same random seed producing the same simulations.
    sens_array.get_error_integrator().reseed_error_chain()
    sens_array.get_field().set_sim_data(sim_data)

    meas = sens_array.sim_measurements()
    
    sim_exp = {(sim_key,sens_key,exp_keys.meas):meas,}
               
    # RETURN: dict[str,np.array.shape=(n_sens,n_comps,n_time_steps)]
    return sim_exp


def _run_all_sims_save_errs(sim_key: str,
                            sens_key: str,
                            sim_data: mh.SimData,
                            sens_array: ISensorArray,
                            num_exp: int,
                            exp_keys: ExpSimKeys,
                            ) -> dict[tuple[str,...],np.ndarray]:
    """Parallelisation helper function for running N simulated experiments on a 
    single worker. This version saves the systematic and random error arrays as
    well as the measurement array.

    Parameters
    ----------
    sim_key : str
        String key identifying the input physics simulation.
    sens_key : str
        String key identifying the sensor array.
    sim_data : mh.SimData
        Simulation data object for the simulation.
    sens_array : ISensorArray
        Sensor array object to apply to the simulation.
    num_exp : int
        Number of experiment 'N' to run on this worker.
    exp_keys : ExpSimKeys
        Contains keys for identifying simulated experiment outputs.

    Returns
    -------
    dict[tuple[str,...],np.ndarray]
        Virtual experiment output data dictionary for the N simulations.
    """
    # NOTE: need to reseed the error chain otherwise each worker inherits the
    # same random seed producing the same simulations.
    sens_array.get_error_integrator().reseed_error_chain()
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

    # RETURN: dict[tuple[str,str,str],
    # np.array.shape=(n_exps,n_sens,n_comps,n_time_steps)]
    return sim_exp


def _run_all_sims_meas_only(sim_key: str,
                            sens_key: str,
                            sim_data: mh.SimData,
                            sens_array: ISensorArray,
                            num_exp: int,
                            exp_keys: ExpSimKeys,
                            ) -> dict[tuple[str,...],np.ndarray]:
    """Parallelisation helper function for running N simulated experiments on a 
    single worker. This version only saves the measurement array.

    Parameters
    ----------
    sim_key : str
        String key identifying the input physics simulation.
    sens_key : str
        String key identifying the sensor array.
    sim_data : mh.SimData
        Simulation data object for the simulation.
    sens_array : ISensorArray
        Sensor array object to apply to the simulation.
    num_exp : int
        Number of experiment 'N' to run on this worker.
    exp_opts : ExpSimKeys
        Contains keys for identifying simulated experiment outputs.

    Returns
    -------
    dict[tuple[str,...],np.ndarray]
        Virtual experiment output data dictionary for the N simulations.
    """
    # NOTE: need to reseed the error chain otherwise each worker inherits the
    # same random seed producing the same simulations.
    sens_array.get_error_integrator().reseed_error_chain()
    sens_array.get_field().set_sim_data(sim_data)

    exp_shape = (num_exp,)+sens_array.get_measurement_shape()

    meas = np.empty(exp_shape,dtype=np.float64)

    for ee in range(num_exp):
        meas[ee,:,:,:] = sens_array.sim_measurements()

    sim_exp = {(sim_key,sens_key,exp_keys.meas):meas,}

    # RETURN: dict[tuple[str,str,str],
    # np.array.shape=(n_exps,n_sens,n_comps,n_time_steps)]
    return sim_exp
