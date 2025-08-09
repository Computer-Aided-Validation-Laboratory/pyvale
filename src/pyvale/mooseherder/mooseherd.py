# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================
import os
import time
import multiprocessing as mp
from pathlib import Path
from multiprocessing.pool import Pool

from pyvale.mooseherder.directorymanager import DirectoryManager
from pyvale.mooseherder.simrunner import SimRunner
from pyvale.mooseherder.inputmodifier import InputModifier


class MooseHerdError(Exception):
    """MooseHerdError: custom error class for flagging errors with the moose
    herd.
    """

class MooseHerd:
    """MooseHerd class that can run parametric sweeps of simulation chains in
    parallel with configurable parallelisation options. Takes a list of
    SimRunner objects and a corresponding list of InputModifiers to insert the
    variables into the input scripts for the SimRunners. When calling run_* the
    herd will first call all InputModifiers in the specified order and then
    call run on all the SimRunners in order. Uses the DirectoryManager class to
    create/clear and log the directories in which each parallel worker is
    creating input files and running simulations.
    """

    __slots__ = ("_runners","_modifiers","_dir_manager","_n_para_sims",
                 "_input_names","_keep_all","_var_sweep","_sweep_iter",
                "_sweep_run_time","_sim_iter","_iter_run_time")

    def __init__(self, sim_runners: list[SimRunner],
                 input_mods: list[InputModifier],
                 dir_manager: DirectoryManager,
                 num_para_sims: int = 1) -> None:
        """
        Parameters
        ----------
        sim_runners : list[SimRunner]
            list of objects that inherit from the SimRunner ABC in the order
            they need to be run. The mesh needs to be created before runnning
            moose so a common chain would be [GmshRunner,MooseRunner].
        input_mods : list[InputModifier]
            list of InputModifiers to create the required input scripts for
            the SimRunners.
        dir_manager : DirectoryManager
            Used to control how many and which directories are used to run the
            simulations.
        num_para_sims : int, optional
            Number of simulations to run in parallel, by default 1
        """

        self._runners = sim_runners
        self._modifiers = input_mods
        self._dir_manager = dir_manager

        self._n_para_sims = num_para_sims

        self._input_names = [f'sim-{ii+1}' for ii,_ in enumerate(sim_runners)]

        self._keep_all: bool = True

        self._var_sweep = list([])

        self._sweep_iter: int = 0
        self._sweep_run_time: float = -1.0

        self._sim_iter: int = 0
        self._iter_run_time: float = -1.0




    def set_input_copy_names(self, input_names: list[str] | None = None) -> None:
        """set_input_copy_name: sets the name that will be used when copying
        input files to the working directories for the sweep. The defualt name
        is 'sim-i' so the first combination of variables in the simulation chain
        will be called 'sim-1-1'.

        Parameters
        ----------
        input_names : list[str] | None
            List of name prefixes to be used for the simulation files. Defaults
            to None.

        Returns
        -------

        Raises
        ------
        MooseHerdError
            The lengths of the sim runner list and the input
            modifier lists are not the same.

        """
        if input_names is None:
            self._input_names = [f'sim-{ii+1}' for ii,_ in enumerate(self._runners)]
            return

        if len(input_names) != len(self._runners):
            raise MooseHerdError(f'The length of the input names ({len(input_names)})'
                                 'must match the length of the sim runners ' +
                                 f'and input modifiers ({len(self._runners)})')

        self._input_names = input_names


    def set_keep_flag(self, keep_all: bool = True) -> None:
        """set_keep_flag: flag used for allowing multiple calls to run_para or
        run_seq to keep everything or to overwrite with every call to run_*.

        Parameters
        ----------
        keep_all : bool
            True = keep all inputs and outputs with
            multiple calls to run_*. False = overwrite inputs and outputs
            with multiple calls to run_*. Defaults to True.

        Returns
        -------

        """
        self._keep_all = keep_all


    def set_num_para_sims(self, n_para: int = 1) -> None:
        """set_num_para_sims: sets the number of simulation chains to run in
        parallel. Limits the number

        Parameters
        ----------
        n_para : int
            Number of parallel simulation to run. Defaults to 1.

        Returns
        -------

        """
        n_para = int(n_para)
        n_cpus = os.cpu_count()

        if n_cpus is None:
            pass
        elif n_para <= 0:
            n_para = 1
        elif n_para > n_cpus:
            n_para = n_cpus

        if self._n_para_sims != n_para:
            self._n_para_sims = n_para


    def get_sim_iter(self) -> int:
        """get_sim_iter: returns the current simulation iteration corresponding
        to the combination of variables being analysed. This number will
        accumulate with multiple calls to run_* is keep_all=true.

        Parameters
        ----------

        Returns
        -------
        int
            current simulation iteration number.

        """
        return self._sim_iter


    def get_sweep_iter(self) -> int:
        """get_sweep_iter: gets the current sweep iteration. The sweep
        iteration is incremented with every call to run_* if keep_all = true.
        If keep_all = false then sweep_iter is held at 1.

        Parameters
        ----------

        Returns
        -------
        int
            current sweep iteration number.

        """
        return self._sweep_iter


    def reset_iter_counts(self) -> None:
        """reset_iter_counts: resets the simulation iteration and the sweep
        iteration counters to zero.

        Parameters
        ----------

        Returns
        -------

        """
        self._sim_iter = 0
        self._sweep_iter = 0


    def _get_process_name(self) -> str:
        """_get_process_name: only here for monkey patching with pytest.

        Parameters
        ----------

        Returns
        -------
        str
            the process name string.

        """
        return mp.current_process().name


    def _get_worker_num(self) -> str:
        """_get_worker_num: helper function to get the worker number for the
        current sub-process.

        Parameters
        ----------

        Returns
        -------
        str
            number string taken from the process name. If this is the main
            process returns '1'.

        """
        name = self._get_process_name()

        if name == 'MainProcess':
            worker_num = '1'
        else:
            worker_num = name.split('-',1)[1]

        if int(worker_num) > self._n_para_sims:
            worker_num = str((int(worker_num) % self._n_para_sims)+1)

        return worker_num


    def _get_run_num(self, sim_iter: int, worker_num: str) -> str:
        """_get_run_num: helper function to get the run directory number string

        Parameters
        ----------
        sim_iter : int
            the current simulation iteration.
        worker_num : str
            the worker number extracted from the current
            process number as a string.

        Returns
        -------
        str
            the string specifying the run directory number for this
            simulation iteration.

        """
        if self._keep_all:
            run_num = str(sim_iter+1)
        else:
            run_num = worker_num

        return run_num


    def _mod_input(self,
                   modifier: InputModifier,
                   mod_vars: dict | None,
                   save_file: Path) -> None:
        """_mod_input: helper function that uses the input modifier to write
        new variables to the input file and save it to the specified path.

        Parameters
        ----------
        modifier : InputModifier
            input modifier for the specified type of
            input file.
        mod_vars : dict | None
            dictionary of variables to write to the
            input file, if None just copy the input file.
        save_file : Path
            path with file name and extension to output the
            modified input file.

        Returns
        -------

        """
        if mod_vars is not None:
            modifier.update_vars(mod_vars)
        modifier.write_file(save_file)


    def _run(self, runner: SimRunner, run_file: Path) -> Path | None:
        """_run: helper function to call the SimRunner and get the path to the
        output file.

        Parameters
        ----------
        runner : SimRunner
            for running the simulation, must be a class
            that implements the SimRunner ABC.
        run_file : Path
            path to the input file to run with SimRunner.


        Returns
        -------
        Path | None
            Path to the output or None

        """
        runner.run(run_file)
        return runner.get_output_path()


    def run_once(self, sim_iter: int, var_list: list[dict | None]
                 ) -> list[Path | None]:
        """run_once: runs a specific simulation chain with the given variable
        list once and returns a list of paths to the output files. Used by
        run_seq and run_para for parallelisation.

        Parameters
        ----------
        sim_iter : int
            current simulation iteration which is the index of
            the var_list from the var_sweep.
        var_list : list[dict  |  None]
            list of dictionaries that contain
            the variables that will be run for this iteration.

        Returns
        -------
        list[Path | None]
            list of paths to the simulation output. If there
            is no useful output from the runner in the simulation chain it
            returns None in the list.

        """
        iter_start_time = time.perf_counter()

        worker_num = self._get_worker_num()
        run_dir = self._dir_manager.get_run_dir(int(worker_num)-1)
        run_num = self._get_run_num(sim_iter,worker_num)

        run_files = list([])
        for ii,mm in enumerate(self._modifiers):
            ext = mm.get_input_file().suffix
            run_files.append(run_dir / (self._input_names[ii] +'-'+run_num+ext))
            self._mod_input(mm,var_list[ii],run_files[ii])

        output_list = list([])
        for ii,rr in enumerate(self._runners):
            output_list.append(self._run(rr,run_files[ii]))

        self._iter_run_time = time.perf_counter() - iter_start_time

        return output_list


    def _start_sweep(self, var_sweep: list[list[dict | None]]) -> float:
        """_start_sweep: helper function used at the start of a variable sweep
        in either run_seq or run_para. Sets the var_sweep attribute, deals with
        the management of directories and starts the performance counter.

        Parameters
        ----------
        var_sweep : list[list[dict  |  None]]
            as passed to run_seq/para.

        Returns
        -------
        float
            performance timer start value.

        """
        self._var_sweep = var_sweep

        if not self._keep_all:
            self.reset_iter_counts()
            self._dir_manager.clear_dirs()
            self._dir_manager.create_dirs()

        return time.perf_counter()


    def _end_sweep(self, start_sweep_time: float,
                   output_files: list[list[Path | None]]) -> None:
        """_end_sweep: helper function called at the end of runseq/para.
        Reacords the sweep run time. Increments the iteration counters. and
        writes the output key and sweep variables to the first workers
        directory.

        Parameters
        ----------
        start_sweep_time : float
            the sweep start time taken from the
            _start_sweep() function.
        output_files : list[list[Path]]
            list of list of paths to the
            simulation chain output files.


        Returns
        -------

        """
        self._sweep_run_time = time.perf_counter() - start_sweep_time

        self._sweep_iter += 1
        self._sim_iter += len(self._var_sweep)

        self._dir_manager.set_output_paths(output_files)
        self._dir_manager.write_output_key(self._sweep_iter)
        self._dir_manager.write_sweep_vars(self._var_sweep,self._sweep_iter)


    def run_sequential(self, var_sweep: list[list[dict | None]]
                       ) -> list[list[Path | None]]:
        """run_sequential: runs the variable sweep given in var_sweep
        sequentially and returns the paths to the simulation outputs.

        Parameters
        ----------
        var_sweep : list[list[dict | None]]
            outer list is the simulation
            iteration, inner list is the position in the simulation chain
            that the variable dictionary corresponds to. The dictionary
            contains the variables that will be inserted into the input
            file before calling run on the SimRunner. If None instead of
            a dictionary then the input file is copied with no modification


        Returns
        -------
        list[list[Path | None]]
            outer list is the simulation iteration and
            the inner list corresponds to the position of the SimRunner in
            the cimulation chain. Gives the path to the simulation output
            or None if no useful output is produced.

        """
        start_sweep_time = self._start_sweep(var_sweep)

        output_files = list([])

        ii = self._sim_iter
        for vv in var_sweep:
            output_files.append(self.run_once(ii,vv))
            ii += 1


        self._end_sweep(start_sweep_time,output_files)

        return output_files


    def run_para(self, var_sweep: list[list[dict | None]]
                 ) -> list[list[Path | None]]:
        """run_para: runs the variable sweep with the simulation chain in
        parallel.

        Parameters
        ----------
        var_sweep : list[list[dict | None]]
            outer list is the simulation
            iteration, inner list is the position in the simulation chain
            that the variable dictionary corresponds to. The dictionary
            contains the variables that will be inserted into the input
            file before calling run on the SimRunner. If None instead of
            a dictionary then the input file is copied with no modification


        Returns
        -------
        list[list[Path | None]]
            outer list is the simulation iteration and
            the inner list corresponds to the position of the SimRunner in
            the cimulation chain. Gives the path to the simulation output
            or None if no useful output is produced.

        """
        sweep_start_time = self._start_sweep(var_sweep)

        with Pool(self._n_para_sims) as pool:
            processes = list([])

            ii = self._sim_iter
            for vv in var_sweep:
                processes.append(pool.apply_async(self.run_once, args=(ii,vv)))
                ii += 1

            output_files = [pp.get() for pp in processes]

        self._end_sweep(sweep_start_time, output_files)

        return output_files


    def get_sweep_time(self) -> float:
        """get_sweep_time

        Parameters
        ----------

        Returns
        -------
        float
            the time taken for the variable sweep to run based on the
            performance counter.

        """
        return self._sweep_run_time


    def get_iter_time(self) -> float:
        """get_iter_time

        Returns
            float: the time taken for the current simulation iteration to run.

        Parameters
        ----------

        Returns
        -------

        """
        return self._iter_run_time

