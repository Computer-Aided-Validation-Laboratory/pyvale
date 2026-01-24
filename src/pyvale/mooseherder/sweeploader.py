# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================
import os
import json
from pathlib import Path
from multiprocessing.pool import Pool
from pyvale.mooseherder.directorymanager import DirectoryManager
import pyvale.mooseherder.directorymanager as dm
from pyvale.mooseherder.exodusloader import ExodusLoader
from pyvale.mooseherder.simdata import SimData, SimLoadConfig


class SweepLoader:
    """Used to read the output from one or more calls to mooseherd.run_para().
    has configurable options for reading in the variable sweep in parallel.
    """
    __slots__ = ("_dir_manager","_output_files","_n_para_read")

    def __init__(self,
                 dir_manager: DirectoryManager,
                 num_para_read: int = 1) -> None:
        """Construct the sweep reader with a directory manager and
        the number of simulations to read in parallel.

        Parameters
        ----------
        dir_manager : DirectoryManager
            The directory manager as used by the mooseherd that ran the
            simulation sweep.
        num_para_read : int, optional
            Number of simulation to read in parallel. Defaults to 1.
        """
        self._dir_manager = dir_manager
        self._output_files = list([])
        self._n_para_read = num_para_read


    def read_output_file_key(self, sweep_iter: int) -> list[list[Path | None]]:
        """read_output_key: reads the output key json file produced by running
        the variable sweep. The output key file maps which simulation were run
        in a given sub directory.

        Parameters
        ----------
        sweep_iter : int
            sweep iteration to read. This is the number that
            is appended to the output key file e.g. output-key-2.json for
            the 2nd call to run_seq/para in the mooseherd.


        Returns
        -------
        list[list[Path]]
            paths to the outputs from the variable sweep the
            outer list is the simulation iteration and the inner list is
            the position in the simulation chain.

        """
        output_key = self._dir_manager.get_output_key_file(sweep_iter)
        if not output_key.is_file():
            raise FileNotFoundError(f'Output key file for sweep iteration {sweep_iter} not found at path: {output_key}')

        with open(output_key, 'r', encoding='utf-8') as okf:
            output_files = json.load(okf)

        return dm.output_str_to_paths(output_files)


    def read_all_output_file_keys(self) -> list[list[Path | None]]:
        """read_all_output_keys: as read_output_keys() but finds all output key
        files in the first sub-directory and reads them.

        Parameters
        ----------

        Returns
        -------
        list[list[Path]]
            paths to the outputs from the variable sweep the
            outer list is the simulation iteration and the inner list is
            the position in the simulation chain.

        Raises
        ------
        FileNotFoundError
            No output key files found in the first sub-
            directory.

        """
        output_paths = self._find_files_by_str(self._dir_manager.get_output_key_tag(),
                                               self._dir_manager.get_run_dir(0))

        if len(output_paths) == 0:
            raise FileNotFoundError("No output key json files found.")

        output_files = list([])
        for output_path in output_paths:
            sweep_iter = output_path.name.split('.')[0].split('-')[-1]
            output_files = output_files + self.read_output_file_key(int(sweep_iter))

        self._output_files = output_files
        return self._output_files


    def read_sweep_var_file(self, sweep_iter: int = 1
                            ) -> list[list[dict | None]]:
        """read_sweep_var_file: reads the file containing the sweep variable
        dictionary for a given sweep iteration and returns it.

        Parameters
        ----------
        sweep_iter : int
            The sweep iteration (i.e. the call
            number to mooseher.run_seq/para) Defaults to 1.

        Returns
        -------
        list[list[dict | None]]
            The sweep variables as passed to mooseherd
            .run_seq/para.

        Raises
        ------
        FileNotFoundError
            Sweep file not found.

        """
        sweep_var_file = self._dir_manager.get_sweep_var_file(sweep_iter)
        if not sweep_var_file.is_file():
            raise FileNotFoundError(
                f'Sweep variable file for sweep iteration {sweep_iter}'+
                f'not found at path: {sweep_var_file}')

        with open(sweep_var_file, 'r', encoding='utf-8') as svf:
            sweep_vars = json.load(svf)

        return sweep_vars


    def read_all_sweep_var_files(self) -> list[list[dict | None]]:
        """read_all_sweep_var_files: as read sweep variables but finds all
        sweep variables in the first sub-directory and reads them.

        Parameters
        ----------

        Returns
        -------
        list[list[dict | None]]
            The sweep variables as passed to mooseherd
            .run_seq/para. Additional sweep iterations are appended to the
            end of the list.

        Raises
        ------
        FileNotFoundError
            No sweep variable files found.

        """
        sweep_var_paths = self._find_files_by_str(self._dir_manager.get_sweep_var_tag(),
                                                  self._dir_manager.get_run_dir(0))

        if len(sweep_var_paths) == 0:
            raise FileNotFoundError("No sweep variable json files found.")

        sweep_vars = list([])
        for ii,_ in enumerate(sweep_var_paths):
            sweep_vars = sweep_vars  + self.read_sweep_var_file(ii+1)

        return sweep_vars


    def _find_files_by_str(self, search_str: str, search_path: Path
                           ) -> list[Path]:
        """_find_files_by_str: helper function for finding output key and sweep
        variable files based on a specific string.

        Parameters
        ----------
        search_str : str
            the string to find in the file name.
        search_path : Path
            the path to the directory to search for files
            containing the specified string.


        Returns
        -------
        list[Path]
            list of paths to the files found with the string.

        """
        found_files = list([])

        all_files = os.listdir(search_path)

        for ff in all_files:
            if search_str in ff:
                found_files.append(Path(ff))

        found_files.sort()

        return found_files


    def get_output_files(self) -> list[list[Path | None]]:
        """get_output_files

        Parameters
        ----------

        Returns
        -------
        list[list[Path]]
            returns the paths to the output files as read
            from the output key files.

        """
        return self._output_files


    def read_results_once(self,
                          output_files: list[Path | None],
                          read_config: SimLoadConfig | None = None
                          ) -> list[SimData | None]:
        """read_results_once: reads a specific simulation at the specified
        path based on the specified read configuration. If the read
        configuration is None then read everything.

        Parameters
        ----------
        output_file : list[Path | None] :
            Path to the file to read
        read_config : SimReadConfig | None
            class to specify the data to read

        Returns
        -------
        list[SimData | None]
            list of data classes holding the simulation
            data for each simulation in the chain. Will be None for a pre-
            processor like Gmsh that has not output.

        """
        data_list = list([])

        for ff in output_files:
            if ff is None:
                data_list.append(None)
            else:
                #TODO: replace with output reader ABC
                reader = ExodusLoader(ff)
                if read_config is None:
                    data_list.append(reader.load_all_sim_data())
                else:
                    data_list.append(reader.load_sim_data(read_config))

        return data_list


    def read_sequential(self,
                                sweep_iter: int | None = None,
                                read_config: SimLoadConfig | None = None
                                ) -> list[list[SimData]]:
        """read_results_sequential: reads the variable sweep results
        sequentially. Can read a specific iteration with a specific read config
        but defaults to reading everything found in the simulation directories.

        Parameters
        ----------
        sweep_iter : int | None
            sweep iteration number to read.
            Reads the output key file for this iteration. Defaults to None.
        read_config : SimReadConfig | None
            object for specifying
            which variables are to be extracted from the output. Defaults
            to None.

        Returns
        -------
        list[list[SimData]]
            list of lists of SimData objects containing the
            simulation results corresponding to each combination of
            variables.

        """
        self._start_read_output_keys(sweep_iter)

        sweep_results = list([])
        for ii,ff in enumerate(self._output_files):
            sweep_results.append(self.read_results_once(ff,read_config))

        return sweep_results


    def read_results_para(self,
                          sweep_iter: int | None = None,
                          read_config: SimLoadConfig | None = None
                          ) -> list[list[SimData]]:
        """read_results_para: reads the variable sweep results in parallel
        Can read a specific iteration with a specific read config but defaults
        to reading everything found in the simulation directories.

        Parameters
        ----------
        sweep_iter : int | None
            sweep iteration number to read.
            Reads the output key file for this iteration. Defaults to None.
        read_config : SimReadConfig | None
            object for specifying
            which variables are to be extracted from the output. Defaults
            to None.

        Returns
        -------
        list[list[SimData]]
            list of lists of SimData objects containing the
            simulation results corresponding to each combination of
            variables.

        """
        self._start_read_output_keys(sweep_iter)

        with Pool(self._n_para_read) as pool:
            processes = list([])
            for ff in self._output_files:
                    processes.append(pool.apply_async(
                        self.read_results_once, args=(ff,read_config)))

            sweep_results = [pp.get() for pp in processes]

        return sweep_results


    def _start_read_output_keys(self, sweep_iter: int | None):
        """_start_read: helper function to read the output keys for a specific
        simulation iteration prior to reading the simulation outputs.

        Parameters
        ----------
        sweep_iter : int | None
            sweep iteration number to read. If None
            read all of the output keys.

        Returns
        -------

        """
        if self._output_files == '':
            self._output_files = self.read_output_file_key(sweep_iter=1)

        if sweep_iter is None:
            self.read_all_output_file_keys()


