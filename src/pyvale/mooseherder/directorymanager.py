# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import os
import shutil
import json
from pathlib import Path

class DirectoryManager:
    """Manages directories for running simulations in parallel with the
    mooseherd.
    """
    __slots__ = ("_n_dirs","_sub_dir","_base_dir","_run_dirs","_output_paths",
                 "_output_key_tag","_sweep_var_tag")

    def __init__(self, n_dirs: int = 1) -> None:
        """
        Parameters
        ----------
        n_dirs : int, optional
            Number of working directories to be created, by default 1.
        """

        self._n_dirs = n_dirs
        self._sub_dir = 'sim-workdir'
        self._base_dir = Path().cwd()
        self._run_dirs = self._set_run_dirs()
        self._output_paths = list([])
        self._output_key_tag = 'output-key'
        self._sweep_var_tag = 'sweep-vars'


    def _set_run_dirs(self) -> list[Path]:
        """_set_run_dirs: helper function that populates the list of
        directories that will be created by the manager. Uses the base directory
        at the start of the path and then creates numbered sub-directory paths
        based on the sub directory name and number of directories specified.

        Parameters
        ----------

        Returns
        -------
        list[Path]
            List of paths to the working directories to run simulations in.
        """
        run_dirs = list([])
        for nn in range(self._n_dirs): # type: ignore
            run_dirs.append(self._base_dir / (self._sub_dir + '-' + str(nn+1)))

        return run_dirs


    def set_sub_dir_name(self, sub_dir_name: str) -> None:
        """set_sub_dir_name: used to set the string used at the start of the
        created sub-directores. default on creation is 'sim-workdir'. Populates
        the list of run directories using the new sub directory name.

        Parameters
        ----------
        sub_dir_name : str
            string to be used to name the created
            sub-directories within the base directory.
        sub_dir_name: str :

        """
        self._sub_dir = sub_dir_name
        self._run_dirs = self._set_run_dirs()


    def set_base_dir(self, base_dir: Path, clear_old_dirs = False) -> None:
        """set_base_dir: sets the base directory to create sub-directors for
        running the simulations. The base directory must exist.

        Parameters
        ----------
        base_dir : Path
            directory in which the new working directories will
            be created.
        clear_old_dirs : bool
            deletes previous directories in
            the base directory and their contents if they exist. Defaults
            to False.

        Returns
        -------

        Raises
        ------
        FileExistsError
            the selected base directory does no exist.

        """
        if not base_dir.is_dir():
            raise FileExistsError("Specified base directory does not exist.")

        if clear_old_dirs:
            self.clear_dirs()

        self._base_dir = base_dir
        self._run_dirs = self._set_run_dirs()


    def get_output_key_tag(self) -> str:
        """get_output_key_tag: returns the string used to name the output
        key files that map the simulation outputs to the various directories
        that are being managed.

        Parameters
        ----------

        Returns
        -------
        str
            common string used to name the output key json files.

        """
        return self._output_key_tag


    def get_sweep_var_tag(self) -> str:
        """get_sweep_var_tag: returns ths string used to name the sweep
        variable json file that contains a copy of the dictionary the user
        provided as part of the parameter sweep.

        Parameters
        ----------

        Returns
        -------
        str
            common string used to name the sweep variable json files.

        """
        return self._sweep_var_tag


    def create_dirs(self) -> list[Path]:
        """Creates the specified number of directories based on
        the sub_dir name within the base directory.

        Parameters
        ----------

        Returns
        -------
        list[Path]
            list of paths to the directories to create.

        """
        for rr in self._run_dirs:
            if not rr.is_dir():
                rr.mkdir()

        return self._run_dirs


    def clear_dirs(self) -> None:
        """Deletes all working directories in the base directory
        that have the corresponding sub-directory name and their contents.
        """
        all_dirs = os.listdir(self._base_dir)
        for dd in all_dirs:
            if os.path.isdir(self._base_dir / dd):
                if self._sub_dir in dd:
                    shutil.rmtree(self._base_dir / dd)

    def reset_dirs(self) -> list[Path]:
        """Helper function that first clears the working directories if they
        exist and then creates new working directories ready for a simulation
        sweep.

        Returns
        -------
        list[Path]
            List of paths to the created working directories
        """
        self.clear_dirs()
        return self.create_dirs()

    def get_all_run_dirs(self) -> list[Path]:
        """Returns the list of paths to all the directories
        that can be used to run simulations.

        Parameters
        ----------

        Returns
        -------
        list[Path]
            paths to all created directories.

        """
        return self._run_dirs


    def get_run_dir(self, dir_num: int) -> Path:
        """get_run_dir: returns the path to the run directory based on the
        input directory number. The directory number can be greater than the
        number of managed directories in the run_dirs list in this case the
        directory number will wrap and point at an existing run directory
        allowing multiple simulations to be run in the same directory.

        Parameters
        ----------
        dir_num : int
            number of the directory path to be retrieved. Can be
            greater than the specified number of directories and will wrap
            appropriately


        Returns
        -------
        Path
            path to the directory

        """
        if dir_num < 0:
            dir_num = 0
        elif dir_num >= self._n_dirs:
            dir_num = dir_num % self._n_dirs

        return self._run_dirs[dir_num]


    def set_output_paths(self, output_paths: list[list[Path | None]]) -> None:
        """set_output_paths: sets the list of lists to the simulation output
        based on herder input.

        Parameters
        ----------
        output_paths : list[list[Path | None]]
            paths to all outputs from the
            variable sweep. Outer list is the variable combination run
            inner list is based on the order the simulations were called.


        Returns
        -------

        """
        self._output_paths = output_paths


    def get_output_paths(self) -> list[list[Path | None]]:
        """get_output_paths: returns the list of lists to the simulation output
        files.

        Parameters
        ----------

        Returns
        -------
        list[list[Path | None]]
            paths to all outputs from the variable sweep.
            Outer list is the variable combination run inner list is based
            on the order the simulations were called.

        """
        return self._output_paths


    def get_output_key_file(self, sweep_iter: int = 1) -> Path:
        """get_output_key_file: gets the path to the output key file created
        during the variable sweep mapping directories to given combinations
        of variables that were run.

        Parameters
        ----------
        sweep_iter : int
            number corresponding to the sweep iteration to
            retrieve. Defaults to 1.

        Returns
        -------
        Path
            path to the output key file that maps output paths to the
            combinations of variables in the sweep.

        """
        return self._run_dirs[0] / f'{self._output_key_tag}-{sweep_iter:d}.json'


    def write_output_key(self, sweep_iter: int) -> None:
        """write_output_key: converts the output paths to strings and saves
        them in json format.

        Parameters
        ----------
        sweep_iter : int
            number corresponing to the sweep iteration to
            write. The sweep iteration is used to number the output key
            files.

        Returns
        -------

        """
        str_output = output_paths_to_str(self._output_paths)

        with open(self.get_output_key_file(sweep_iter), "w", encoding='utf-8') as okf:
            json.dump(str_output, okf, indent=4)


    def get_sweep_var_file(self, sweep_iter: int = 1) -> Path:
        """get_sweep_var_file: path to the json file which contains the
        dictionary of variables that were analysed at the given sweep iteration

        Parameters
        ----------
        sweep_iter : int
            Sweep iteration (call number to herd
            run_para). Defaults to 1.

        Returns
        -------
        Path
            path to the json sweep variable file.

        """
        return self._run_dirs[0] / f'{self._sweep_var_tag}-{sweep_iter:d}.json'


    def write_sweep_vars(self,
                         sweep_vars: list[list[dict | None]],
                         sweep_iter: int = 1) -> None:
        """write_sweep_vars: writes the sweep variable dictionary to a json
        file to log the variables used for each simulation.

        Parameters
        ----------
        sweep_vars : list[list[dict | None]]
            sweep variables as passed to the herd to run.
        sweep_iter : int
            iteration number for number of calls to the
            herd. Defaults to 1.

        Returns
        -------

        """
        with open(self.get_sweep_var_file(sweep_iter), "w", encoding='utf-8') as okf:
            json.dump(sweep_vars, okf, indent=4)



def output_paths_to_str(output_files: list[list[Path | None]]
                        ) -> list[list[str | None]]:
    """output_paths_to_str: helper function for converting the output paths
    to strings to allow them to be saved as json.

    Parameters
    ----------
    output_files : list[list[Path | None]]


    Returns
    -------
    list[list[str]]
        as input with Path converted to str

    """
    str_output = list([])
    for sim_iter in output_files:
        iter_output = list([])
        for output_path in sim_iter:
            if output_path is None:
                iter_output.append(None)
            else:
                iter_output.append(str(output_path))

        str_output.append(iter_output)

    return str_output


def output_str_to_paths(output_files: list[list[str | None]]
                        ) -> list[list[Path | None]]:
    """output_str_to_paths: helper function to convert strings read from output
    key json to paths.

    Parameters
    ----------
    output_files : list[list[str]]
        output file list of path strings as in the output key file.

    Returns
    -------
    list[list[Path]]
        as input with str converted to Path.

    """
    str_output = list([])

    for sim_iter in output_files:
        iter_output = list([])
        for output_path in sim_iter:
            if output_path is None:
                iter_output.append(None)
            else:
                iter_output.append(Path(output_path))

        str_output.append(iter_output)

    return str_output
