# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================
import os
import subprocess
from pathlib import Path
from pyvale.mooseherder.simrunner import SimRunner
from pyvale.mooseherder.mooseconfig import MooseConfig


class MooseRunner(SimRunner):
    """Used to run MOOSE models (*.i) from python."""

    __slots__ = ("_config","_n_threads","_n_tasks","_redirect_stdout",
                 "_arg_list","_input_path")

    def __init__(self, config: MooseConfig):
        """Constructor for MOOSE runner taking a MooseConfig object
        that contains the paths to the main MOOSE install, the MOOSE app and
        the MOOSE app name. Sets default parallelisation options to 1 MPI task
        and 1 thread. Sets environment variables required for MPI setup.


        Parameters
        ----------
        config : MooseConfig
            moose configuration object containing the required paths and app
            name to construct the command string.
        """
        self._config = config.get_config()

        self._n_threads = 1
        self._n_tasks = 1
        self._redirect_stdout = True
        self._arg_list = list('')
        self._input_path = None


    def set_env_vars(self) -> None:
        """Sets environment variables for calling MOOSE with MPI."""
        os.environ['CC'] = 'mpicc'
        os.environ['CXX'] = 'mpicxx'
        os.environ['F90'] = 'mpif90'
        os.environ['F77'] = 'mpif77'
        os.environ['FC'] = 'mpif90'
        os.environ['MOOSE_DIR'] = str(self._config['main_path'])
        if str(self._config['app_path']) not in os.environ["PATH"]:
            os.environ["PATH"] = os.environ["PATH"] + ':' + str(self._config['app_path'])

    def set_threads(self, n_threads: int) -> None:
        """Sets the number of threads asked of MOOSE on the command line.

        Parameters
        ----------
        n_threads : int
            Number of threads.
        """
        # Need to make sure number is sensible based on cpu
        if n_threads <= 0:
            n_threads = 1
        elif os.cpu_count() is None:
            n_threads = 1
        elif n_threads > os.cpu_count(): # type: ignore
            n_threads = os.cpu_count() # type: ignore

        self._n_threads = int(n_threads)

    def set_tasks(self, n_tasks: int) -> None:
        """Sets the number of MPI tasks asked of MOOSE on the command line.

        Parameters
        ----------
        n_tasks : int
            Number of mpi tasks.

        Returns
        -------

        """
        # Need to make sure is sensible based on cpu
        if n_tasks <= 0:
            n_tasks = 1
        elif os.cpu_count() is None:
            n_tasks = 1
        elif n_tasks > os.cpu_count(): # type: ignore
            n_tasks = os.cpu_count() # type: ignore

        self._n_tasks = int(n_tasks)

    def set_stdout(self, redirect_flag: bool = True) -> None:
        """Sets MOOSE to redirect output (True) to file instead of console (False).

        Parameters
        ----------
        redirect_flag : bool
            True = output to stdout file, False
            = output to console. Defaults to True.

        Returns
        -------

        """
        self._redirect_stdout = redirect_flag

    def set_run_opts(self, n_tasks: int = 1,
                    n_threads: int = 1,
                    redirect_out: bool = True) -> None:
        """Sets all options for MOOSE run parallelisation and output.

        Parameters
        ----------
        n_tasks : int
            Number of mpi tasks for MOOSE run.
            Defaults to 1.
        n_threads : int
            Number of threads for MOOSE run.
            Defaults to 1.
        redirect : bool
            Redirect MOOSE output from console to
            file (True). Defaults to False.

        Returns
        -------

        """
        self.set_threads(n_threads)
        self.set_tasks(n_tasks)
        self.set_stdout(redirect_out)


    def get_input_file(self) -> Path | None:
        """get_input_file

        Parameters
        ----------

        Returns
        -------
        Path | None
            full path to the input file or None if not specified.

        """
        return self._input_path


    def set_input_file(self, input_path: Path) -> None:
        """Sets the path to the MOOSE input file and checks it exists.

        Parameters
        ----------
        input_file : Path
            full path and name of *.i MOOSE input script.

        Returns
        -------

        Raises
        ------
        FileNotFoundError
            the MOOSE input script doesn't exist

        """
        if not input_path.is_file():
            raise FileNotFoundError("Input file does not exist.")

        self._input_path = input_path
        self.assemble_arg_list()

    def get_input_dir(self) -> Path | None:
        """Gets the path to the directory for the specified input file.

        Parameters
        ----------

        Returns
        -------
        Path
            path to input file directory, if no input file is specified
            returns None.

        """
        if self._input_path is None:
            return None

        return self._input_path.parent # type: ignore

    def get_input_tag(self) -> str:
        """Gets the input file name string without the path or the .i

        Parameters
        ----------

        Returns
        -------
        str
            input file string, if no input file is specified returns an
            empty string.

        """
        if self._input_path is None:
            return ""

        return self._input_path.stem # type: ignore


    def get_output_path(self) -> Path | None:
        """Gets the file and path for the output exodus file based
        on the specified input file. Includes '_out.e'.

        Parameters
        ----------

        Returns
        -------
        Path
            output exodus file name with path, returns an empty None
            if no input file is specified.

        """
        if self._input_path is None:
            return None

        return self._input_path.parent / (self._input_path.stem +'_out.e')


    def get_arg_list(self) -> list[str]:
        """Run string getter.

        Parameters
        ----------

        Returns
        -------
        str
            command line string to run MOOSE.

        """
        return self._arg_list

    def assemble_arg_list(self, input_file = None) -> list[str]:
        """Assmebles the command line string to run MOOSE based on current
        options.

        Parameters
        ----------
        input_file : str
            Full path to MOOSE input file, if not
            empty updates the input file. Defaults to "".

        Returns
        -------
        str
            command line string that will be used by the runner when run
            is called.

        """
        if input_file is not None:
            self.set_input_file(input_file)

        if self._input_path is None:
            raise RuntimeError('No input file specified, set one using set_input_file or by passing on into this function.')

        arg_list = []
        if self._n_tasks > 1:
            arg_list = ['mpirun','-np',str(self._n_tasks)]

        arg_list = arg_list + [str(self._config['app_name']) \
                    ,f'--n-threads={self._n_threads}','-i' \
                    ,str(self._input_path.name)]

        if self._redirect_stdout:
            arg_list = arg_list + ['--redirect-stdout']

        self._arg_list = arg_list

        return self._arg_list


    def run(self, input_file = None) -> None:
        """Runs MOOSE based on current options by passing run string to
        subprocess shell.

        Parameters
        ----------
        input_file : Path
            Full path to MOOSE input file, if not
            empty updates the input file. Defaults to None.

        Returns
        -------

        """
        if input_file is not None:
            self.set_input_file(input_file)

        if self._input_path is None:
            raise RuntimeError("Set input path before calling run.")

        self.set_env_vars()

        self.assemble_arg_list()
        subprocess.run(self._arg_list,
                       shell=False,
                       cwd=str(self._input_path.parent),
                       check=False)
