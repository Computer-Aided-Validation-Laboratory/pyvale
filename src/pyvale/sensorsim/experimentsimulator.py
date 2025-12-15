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
    """Parallelisation strategy to use for simulated experiments.
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
class ExpSimSaveKeys:
    """Default string keys used for saving experiment simulation data in the
    returned data dictionary. These keys appear in the last position of the
    tuple key and indicate the data that is available. The first two positions
    in the tuple key indicate the simulation key and sensor array key. For
    example (sim_key,sensor_key,data_key). Setting any of the data keys in this
    class to None will mean that data is not saved from the simulation. This is
    useful if you only want the simulated measurements but not the error
    breakdown or if you are noy using field errors and don't need the perturbed
    sensor data. Note that the measurement key must be specified.
    """

    meas: str = "meas"
    """Default string key for the measurement array, must be specified.
    """

    sens_times: str = "sens_times"
    """Default string key for the nominal sensor sampling times. Must be
    specified and stored such that resuls traces can be plotted.
    """

    sys: str | None = "sys_errs"
    """Deafult string key for the systematic error array. Set to None to not
    save the array.
    """

    rand: str | None = "rand_errs"
    """Deafult string key for the random error array. Set to None to not
    save the array.
    """

    truth: str | None = None
    """String key for the truth array. By default None as the truth array can
    be calculated from the measurement array and error arrays.
    """

    pert_sens_times: str | None = "pert_sens_times"
    """Default string key for perturbed sensor times when field errors are used.
    Set to None if there are no field errors or if the field error is not
    perturbing the sensor sampling times.
    """

    pert_sens_pos: str | None = "pert_sens_pos"
    """Default string key for perturbed sensor positions when field errors are
    used. Set to None if there are no field errors or if the field error is not
    perturbing the sensor positions.
    """

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


class ExperimentSimulator:
    """An experiment simulator for running Monte-Carlo simulation by applying a
    dictionary of sensor arrays to a dictionary of simulations over a given
    number of user defined experiments.
    """
    __slots__ = ("_sim_dict","_sens_dict","_exp_sim_opts","_save_keys")

    def __init__(self,
                 sim_dict: dict[str,mh.SimData],
                 sensor_arrays: dict[str,ISensorArray],
                 exp_sim_opts: ExpSimOpts | None = None,
                 exp_save_keys: ExpSimSaveKeys | None = None,
                 ) -> None:
        """
        Parameters
        ----------
        sim_dict : dict[str,mh.SimData]
            The simulations
        sensor_arrays : dict[str,ISensorArray]
            The sensor arrays that will be applied to each simulation to
            generate the virtual experiment data.

        """

        self._sim_dict = sim_dict
        self._sens_dict = sensor_arrays

        if exp_sim_opts is None:
            self._exp_sim_opts = ExpSimOpts()
        else:
            self._exp_sim_opts = exp_sim_opts

        if exp_save_keys is None:
            self._save_keys = ExpSimSaveKeys()
        else:
            self._save_keys = exp_save_keys


    def get_exp_save_keys(self) -> ExpSimSaveKeys:
        """Gets the experiment simulation data keys.

        Returns
        -------
        ExpSimSaveKeys
            Dataclass containing the keys used to identify output data from the
            simulated experiments.
        """
        return self._save_keys

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
                        ) -> dict[tuple[str,...],np.ndarray]:
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
            simulation. See the `ExpSimSaveKeys` dataclass for valid data keys.
            The 'measurement 'data arrays returned for simulated experiment
            output have shape=(n_exps,n_sens,n_comps,n_time_steps).

        Raises
        ------
        ExpSimError
            The number of virtual experiments to run is not a positive integer.
        """

        #-----------------------------------------------------------------------
        # Setup by checking and setting the number of experiments
        if num_exp_per_sim <= 0:
            raise ExpSimError(
                "Number of experiments per sim must be a positive integer"
            )

        #-----------------------------------------------------------------------
        # Build function call list and associated keys for the simulation
        # NOTE: avoids if/else in the hot loop and only calls the required
        # functions using a loop over the list and getattr() on the sens_array
        # or perturbed sens_data objects.
        exp_keys = self._save_keys
        sens_funcs: list[tuple[str,str]] = [(exp_keys.meas,"sim_measurements"),]

        if exp_keys.rand is not None:
            sens_funcs.append((exp_keys.rand,"get_errors_random"))

        if exp_keys.sys is not None:
            sens_funcs.append((exp_keys.sys,"get_errors_systematic"))

        if exp_keys.truth is not None:
            sens_funcs.append((exp_keys.truth,"get_truth"))

        sens_vars = []
        if exp_keys.pert_sens_times is not None:
            sens_vars.append((exp_keys.pert_sens_times,"sample_times"))

        if exp_keys.pert_sens_pos is not None:
            sens_vars.append((exp_keys.pert_sens_pos,"positions"))

        #-----------------------------------------------------------------------
        # 1) para over sim_data/sens_array, run N per worker
        if (self._exp_sim_opts.workers is not None and
            self._exp_sim_opts.para == EExpSimPara.ALL):

            exp_data = self._run_para_all(num_exp_per_sim,sens_funcs,sens_vars)

        # 2) para over all sim_data/sens_array/Nsims
        elif (self._exp_sim_opts.workers is not None and
              self._exp_sim_opts.para == EExpSimPara.SPLIT):

            exp_data = self._run_para_split(num_exp_per_sim,
                                            sens_funcs,
                                            sens_vars)

        # 3) Run everything sequentially
        else:
            exp_data = self._run_sequential(num_exp_per_sim,
                                            sens_funcs,
                                            sens_vars)

        #-----------------------------------------------------------------------
        # dict[tuple[str,...],shape=(n_sims,n_exps,n_sens,n_comps,n_time_steps)]
        return exp_data

    #---------------------------------------------------------------------------
    def _run_para_all(self,
                      num_exp: int,
                      sens_funcs: list[tuple[str,str]],
                      sens_vars: list[tuple[str,str]],
                      ) -> dict[tuple[str,...],np.ndarray]:
        """Runs virtual experiments in parallel based on the `ALL` strategy where 
        `num_exp` experiments are run on each worker.
        

        Parameters
        ----------
        num_exp : int
            Number of virtual experiments to run per combination of simulation
            data and sensor array.
        sens_funcs : list[tuple[str,str]]
            The string keys and string names of functions to call on the sensor 
            array as part of the simulation. The first tuple should always call 
            the `sim_measurements()` function and has default key 'meas'.
        sens_vars : list[tuple[str,str]]
            The string keys and string names for the variables to extract from
            the perturbed sensor data, useful for extracting information from
            field errors.

        Returns
        -------
        dict[tuple[str,...],np.ndarray]
            Dictionary of virtual experimental data arrays where the key is a
            tuple with form (sim_key,sens_key,data_key). All data arrays are 
            numpy arrays.
        """
        assert self._exp_sim_opts.workers > 0, ("Number of workers must"
                                + " be greater than 0.")

        time_str_key = self._save_keys.sens_times

        exp_data = {}

        with Pool(self._exp_sim_opts.workers) as pool:
            processes = []
            for (key_sim, sim_data), (key_sens, sens_array) in (
                product(self._sim_dict.items(), self._sens_dict.items())
            ):
                time_key = (key_sim,key_sens,time_str_key)
                exp_data[time_key] = sens_array.get_sample_times()

                args = (
                    num_exp,
                    key_sim,
                    key_sens,
                    sim_data,
                    sens_array,
                    sens_funcs,
                    sens_vars,
                )

                process = pool.apply_async(_run_all_sims,args=args)

                processes.append(process)

            for pp in processes:
                # dict[tuple[str,str,str],
                #      shape=(n_exps,n_sens,n_comps,n_time_steps)]
                sim_exps = pp.get()
                exp_data.update(sim_exps)

        return exp_data

    #---------------------------------------------------------------------------
    def _run_para_split(self,
                        num_exp: int,
                        sens_funcs: list[tuple[str,str]],
                        sens_vars: list[tuple[str,str]],
                        ) -> dict[tuple[str,...],np.ndarray]:
        """Runs virtual experiments in parallel based on the `SPLIT` strategy 
        where one simulated experiment is run on each work splitting the 
        `num_exp` experiments across all workes. Useful for computationally 
        heavy simulation chains.
        

        Parameters
        ----------
        num_exp : int
            Number of virtual experiments to run per combination of simulation
            data and sensor array.
        sens_funcs : list[tuple[str,str]]
            The string keys and string names of functions to call on the sensor 
            array as part of the simulation. The first tuple should always call 
            the `sim_measurements()` function and has default key 'meas'.
        sens_vars : list[tuple[str,str]]
            The string keys and string names for the variables to extract from
            the perturbed sensor data, useful for extracting information from
            field errors.

        Returns
        -------
        dict[tuple[str,...],np.ndarray]
            Dictionary of virtual experimental data arrays where the key is a
            tuple with form (sim_key,sens_key,data_key). All data arrays are 
            numpy arrays.
        """
        assert self._exp_sim_opts.workers > 0, ("Number of workers must"
                                   + " be greater than 0.")

        exp_keys = self._save_keys
        time_str_key = exp_keys.sens_times

        exp_data = {}
        # We are going to have to populate the experiment array on the fly
        # so we need to pre-alloc to index into it as we get results.
        for (key_sim, sim_data), (key_sens, sens_array) in (
            product(self._sim_dict.items(), self._sens_dict.items())
        ):
            # Pre-alloc numpy arrays for simulated measurements.
            exp_shape = (num_exp,) + sens_array.get_measurement_shape()
            for kk, mm in sens_funcs:
                exp_data[(key_sim,key_sens,kk)] = (
                    np.empty(exp_shape,dtype=np.float64)
                )

            # Pre-alloc numpy arrays for perturbed sensor data
            err_int = sens_array.get_error_integrator()
            init_sens_data = err_int.get_sens_data_accumulated()
            for kk, vv in sens_vars:
                attr = getattr(init_sens_data,vv)
                if attr is not None:
                    shape = (num_exp,) + attr.shape
                    exp_data[(key_sim,key_sens,kk)] = (
                        np.empty(shape,dtype=np.float64)
                    )

        with Pool(self._exp_sim_opts.workers) as pool:
            processes_with_id = []
            for (key_sim, sim_data), (key_sens, sens_array) in (
               product(self._sim_dict.items(), self._sens_dict.items())
            ):
                time_key = (key_sim,key_sens,time_str_key)
                exp_data[time_key] = sens_array.get_sample_times()

                for ee in range(num_exp):

                    args = (key_sim,
                            key_sens,
                            sim_data,
                            sens_array,
                            sens_funcs,
                            sens_vars)

                    process = pool.apply_async(_run_one_sim,args=args)

                    processes_with_id.append({"process":process,
                                              "exp_ind":ee})

            for pp in processes_with_id:
                # dict[tuple[str,..],shape=(n_sens,n_comps,n_time_steps)]
                one_exp_dict = pp["process"].get()
                exp_i = pp["exp_ind"]

                for kk,aa in one_exp_dict.items():
                    # NOTE: broadcast here because general measurement type
                    # arrays are 4D, other arrays like sensor perturbations are
                    # different.
                    # shape=(n_exps,n_sens,n_comps,n_time_steps)
                    exp_data[kk][exp_i] = aa

        return exp_data

    #---------------------------------------------------------------------------
    def _run_sequential(self,
                        num_exp: int,
                        sens_funcs: list[tuple[str,str]],
                        sens_vars: list[tuple[str,str]],
                        ) -> dict[tuple[str,...],np.ndarray]:
        """Runs virtual experiments sequentially on a the main python 
        interpreter.
                
        
        Parameters
        ----------
        num_exp : int
            Number of virtual experiments to run per combination of simulation
            data and sensor array.
        sens_funcs : list[tuple[str,str]]
            The string keys and string names of functions to call on the sensor 
            array as part of the simulation. The first tuple should always call 
            the `sim_measurements()` function and has default key 'meas'.
        sens_vars : list[tuple[str,str]]
            The string keys and string names for the variables to extract from
            the perturbed sensor data, useful for extracting information from
            field errors.

        Returns
        -------
        dict[tuple[str,...],np.ndarray]
            Dictionary of virtual experimental data arrays where the key is a
            tuple with form (sim_key,sens_key,data_key). All data arrays are 
            numpy arrays.
        """
        time_str_key = self._save_keys.sens_times

        exp_data = {}

        for (key_sim, sim_data), (key_sens, sens_array) in (
            product(self._sim_dict.items(), self._sens_dict.items())
        ):

            time_key = (key_sim,key_sens,time_str_key)
            exp_data[time_key] = sens_array.get_sample_times()

            exp_res = _run_all_sims(num_exp,
                                    key_sim,
                                    key_sens,
                                    sim_data,
                                    sens_array,
                                    sens_funcs,
                                    sens_vars)

            exp_data.update(exp_res)

        return exp_data

#-------------------------------------------------------------------------------
def _run_one_sim(sim_key: str,
                 sens_key: str,
                 sim_data: mh.SimData,
                 sens_array: ISensorArray,
                 sens_funcs: list[tuple[str,str]],
                 sens_vars: list[tuple[str,str]],
                 ) -> dict[tuple[str,...],np.ndarray]:
    """Helper function for parallelisation of simulated experiments using the 
    `SPLIT` strategy to run a single simulation per worker.

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
    sens_funcs : list[tuple[str,str]]
        String data key and function name as a string in order of execution for
        the simulated experiment.
    sens_vars : list[tuple[str,str]]
        String data key and variable name as a string for the perturbed sensor
        data to extract from the simulation.
        
    Returns
    -------
    dict[tuple[str,...],np.ndarray]
        The simulated experiment data dictionary.
    """
    # NOTE: need to reseed the error chain otherwise each worker inherits the
    # same random seed producing the same simulations.
    err_int = sens_array.get_error_integrator()
    err_int.reseed_error_chain()
    sens_array.get_field().set_sim_data(sim_data)

    sim_exp = {}
    for kk, mm in sens_funcs:
        bound_func = getattr(sens_array,mm)
        sim_exp[(sim_key,sens_key,kk)] = bound_func()

    pert_sens_data = err_int.get_sens_data_accumulated()
    for kk, vv in sens_vars:
        attr = getattr(pert_sens_data,vv)
        if attr is not None:
            sim_exp[(sim_key,sens_key,kk)] = attr

    # RETURN: dict[str,np.array.shape=(n_sens,n_comps,n_time_steps)]
    return sim_exp


def _run_all_sims(num_exp: int,
                  sim_key: str,
                  sens_key: str,
                  sim_data: mh.SimData,
                  sens_array: ISensorArray,
                  sens_funcs: list[tuple[str,str]],
                  sens_vars: list[tuple[str,str]]
                  ) -> dict[tuple[str,...],np.ndarray]:
    """Helper function for parallelisation of virtual experiments using the 
    `ALL` strategy where all `num_exp` experiments are run on a single worker.

    Parameters
    ----------
    num_exp : int
        Number of virtual experiments to run for this combination of simulation
        data and sensor array.
    sim_key : str
        String key identifying the input physics simulation.
    sens_key : str
        String key identifying the sensor array.
    sim_data : mh.SimData
        Simulation data object for the simulation.
    sens_array : ISensorArray
        Sensor array object to apply to the simulation.
    sens_funcs : list[tuple[str,str]]
        String data key and function name as a string in order of execution for
        the simulated experiment.
    sens_vars : list[tuple[str,str]]
        String data key and variable name as a string for the perturbed sensor
        data to extract from the simulation.

    Returns
    -------
    dict[tuple[str,...],np.ndarray]
        The simulated experiment data dictionary.
    """

    # NOTE: need to reseed the error chain otherwise each worker inherits the
    # same random seed producing the same simulations.
    err_int = sens_array.get_error_integrator()
    err_int.reseed_error_chain()
    sens_array.get_field().set_sim_data(sim_data)

    exp_shape = (num_exp,)+sens_array.get_measurement_shape()

    # Get the bound functions for this sensor array once before the sim loop
    # and pre-alloc numpy arrays
    bound_sens_funcs = [] # List of tuples to guarantee execution order
    sim_exp = {}
    for kk, mm in sens_funcs:
        bound_sens_funcs.append((kk,getattr(sens_array,mm)))

        sim_exp[(sim_key,sens_key,kk)] = np.empty(exp_shape,dtype=np.float64)

    # Pre-alloc numpy arrays for perturbed sensor data
    init_sens_data = err_int.get_sens_data_accumulated()
    for kk, vv in sens_vars:
        attr = getattr(init_sens_data,vv)

        if attr is not None:
            shape = (num_exp,) + attr.shape
            sim_exp[(sim_key,sens_key,kk)] = np.empty(shape,dtype=np.float64)

    # Simulation loop, first bound func is always `sim_measurements`
    for ee in range(num_exp):
        for kk, bound_func in bound_sens_funcs:
            # NOTE: array broadcast here, array is normally 4D: [ee,:,:,:] but
            # might be different for storing SensorData arrays
            sim_exp[(sim_key,sens_key,kk)][ee] = bound_func()

        pert_sens_data = err_int.get_sens_data_accumulated()
        for kk, vv in sens_vars:
            attr = getattr(pert_sens_data,vv)
            if attr is not None:
                sim_exp[(sim_key,sens_key,kk)][ee] = attr

    # RETURN: dict[tuple[str,str,str],
    # np.array.shape=(n_exps,n_sens,n_comps,n_time_steps)]
    return sim_exp
