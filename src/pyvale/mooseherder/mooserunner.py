# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Execute MOOSE decks with packaged Rabbit or an explicitly selected app."""

import os
import shutil
import sys
from pathlib import Path

from .simrunner import (
    InputFiles,
    InputValues,
    RunnerBackend,
    RunResult,
    SimRunner,
    _check_backend_exe,
    _check_output_file_name,
    conv_args_dict_to_list,
    execute,
    stage_input_files,
    validate_parameter_name,
)


class MooseRunner(SimRunner):
    """Run ordered MOOSE decks with packaged Rabbit or a custom application.

    Notes
    -----
    The default Rabbit run is serial/SMP. MPI requires both a capable executable
    and a compatible launcher; select the matching solver deck explicitly.
    Each run stages its input tree and writes the configured Exodus output,
    run.log, command.json and
    input_params.json. Existing run directories are rejected to prevent stale output
    from being returned as a successful new simulation.
    """

    def __init__(
        self,
        *,
        moose_exe_file: Path | None = None,
        n_tasks: int | None = None,
        n_threads: int = 1,
        launcher: tuple[str, ...] = ("mpirun", "-np"),
        env_vars: dict[str, str] | None = None,
        timeout: float = 3600.0,
        output_parameter: str = "Outputs/file_base",
        output_file_base: str = "results",
    ) -> None:
        """Configure application selection, parallelism and execution limits.

        Parameters
        ----------
        moose_exe_file : Path | None, optional
            Explicit MOOSE application executable. None selects packaged Rabbit.
        n_tasks : int | None, optional
            MPI rank count. None (the default) or 1 runs without MPI; input_params above
            1 require an MPI-capable application and launcher.
        n_threads : int, optional
            Positive thread count per process. Defaults to 1.
        launcher : tuple[str, ...], optional
            MPI command prefix before the rank count and executable. Defaults to
            ("mpirun", "-np"). Used only when n_tasks is greater than 1.
        env_vars : dict[str, str] | None, optional
            OS environment variable overrides for the solver process, copied at
            construction and merged with the calling Python process's OS
            environment variables (os.environ) when running. For example,
            {"OMP_NUM_THREADS": "4"}. None inherits those OS environment
            variables without overrides. The calling Python process's OS
            environment variables are not modified.
        timeout : float, optional
            Maximum subprocess runtime in seconds. Defaults to 3600.0 (1 hour).
        output_parameter : str, optional
            Native variable controlling the Exodus file base. Defaults to
            "Outputs/file_base"; use an exposed root variable such as "simName" when
            the deck references it. The runner assigns output_file_base.

        output_file_base : str, optional
            Basename for the primary Exodus output, without directories. Defaults
            to "results", producing results.e in the run directory.

        Raises
        ------
        ValueError
            Rank/thread counts, timeout, output parameter or file base are invalid.
        FileNotFoundError
            A custom executable does not exist.
        PermissionError
            A custom executable cannot be executed.
        RuntimeError
            Rabbit or the required MPI launcher/build is unavailable.
        """
        if n_tasks is not None and (type(n_tasks) is not int or n_tasks < 1):
            raise ValueError("n_tasks must be a positive integer.")

        if type(n_threads) is not int or n_threads < 1:
            raise ValueError("n_threads must be a positive integer.")

        if timeout <= 0:
            raise ValueError("timeout must be positive.")

        self.moose_exe_file = (
            moose_exe_file.resolve() if moose_exe_file is not None else None
        )
        self.n_tasks = n_tasks
        self.n_threads = n_threads
        self.launcher = launcher
        self.env_vars = dict(env_vars or {})
        self.timeout = timeout

        validate_parameter_name(output_parameter)
        self.output_parameter = output_parameter
        _check_output_file_name(output_file_base)
        self.output_file_base = output_file_base
        self.get_backend()

    def get_backend(self) -> RunnerBackend:
        """Check the selected executable, Rabbit build and required MPI launcher.

        Returns
        -------
        RunnerBackend
            Resolved application executable. Checks run at construction and before
            every simulation; custom executable MPI capabilities remain user supplied.

        Raises
        ------
        FileNotFoundError
            The selected executable is missing.
        PermissionError
            The executable lacks OS execution permission.
        RuntimeError
            Rabbit cannot be imported, its build lacks requested MPI support, or
            the configured MPI launcher cannot be found.
        """
        if self.moose_exe_file is not None:
            backend = RunnerBackend(_check_backend_exe(self.moose_exe_file))
        else:
            try:
                from rabbit import get_binary_path
                from rabbit.cli import is_mpi_binary
            except (ImportError, OSError) as err:
                raise RuntimeError(f"Rabbit backend is unavailable: {err}") from err
            if self.n_tasks is not None and self.n_tasks > 1 and not is_mpi_binary():
                raise RuntimeError(
                    "MPI requires an MPI-capable Rabbit installation; "
                    "the installed rabbit-fem build is serial/SMP."
                )
            backend = RunnerBackend(_check_backend_exe(get_binary_path()))
        if (
            self.n_tasks is not None
            and self.n_tasks > 1
            and (not self.launcher or shutil.which(self.launcher[0]) is None)
        ):
            raise RuntimeError("MPI launcher is unavailable; configure launcher.")
        return backend

    def run(
        self,
        input_files: InputFiles,
        *,
        output_dir: Path,
        input_params: dict[str, InputValues] | None = None,
        input_root: Path | None = None,
    ) -> RunResult:
        """Stage and solve one MOOSE model with native CLI parameter overrides.

        Parameters
        ----------
        input_files : Path | list[Path] | tuple[Path, ...]
            MOOSE entry deck or ordered list/tuple of decks. Required for this runner.
        output_dir : Path
            New working directory outside the input tree.
        input_params : dict[str, InputValues] | None, optional
            Native input parameter input_params. None uses the model defaults.
        input_root : Path | None, optional
            Input tree to snapshot. Defaults to the parent of a single entry file;
            multiple entry decks require an explicit root.

        Returns
        -------
        RunResult
            Successful primary output, working directory, invocation and log paths.

        Raises
        ------
        ValueError
            Inputs or run configuration are invalid or ambiguous.
        FileNotFoundError
            An input file is missing.
        FileExistsError
            The output directory already exists.
        RuntimeError
            A backend is unavailable, a process fails/times out, or expected output
            is absent or empty.

        Notes
        -----
        Later entry decks override earlier decks; CLI overrides take final precedence.
        The runner manages its output_parameter and Outputs/exodus; do not also supply
        those keys in input_params. Units and array conventions come from the decks.
        """
        input_params = dict(input_params or {})
        # The runner owns the primary output contract; do not silently replace
        # a user's request for a different output location.
        if {
            "Outputs/file_base",
            "Outputs/exodus",
            self.output_parameter,
        } & input_params.keys():
            raise ValueError("The runner manages Outputs/file_base and Outputs/exodus.")

        overrides = conv_args_dict_to_list(input_params)
        backend = self.get_backend()

        command = []

        if self.n_tasks is not None and self.n_tasks > 1:
            command.extend((*self.launcher, str(self.n_tasks)))

        if self.moose_exe_file is None:
            # The public CLI sets Rabbit's library search OS environment variables
            # and checks its build variant. Use this interpreter, never PATH's Python.
            command.extend(
                (sys.executable, "-c", "from rabbit.cli import main; main()")
            )
        else:
            command.append(str(backend.backend_file))

        directory = output_dir.resolve()
        staged = stage_input_files(input_files, directory, input_root)

        command.extend(("-i", *(str(p.relative_to(directory)) for p in staged)))
        command.extend((f"--n-threads={self.n_threads}", *overrides))
        command.extend(
            (f"{self.output_parameter}={self.output_file_base}", "Outputs/exodus=true")
        )

        env_vars = os.environ.copy()
        env_vars.update(self.env_vars)

        log = execute(
            command, directory, input_params, env_vars=env_vars, timeout=self.timeout
        )
        output_file = directory / f"{self.output_file_base}.e"

        if not output_file.is_file() or output_file.stat().st_size == 0:
            raise RuntimeError(f"MOOSE did not produce {output_file}; see {log}")

        return RunResult(output_file, directory, tuple(command), log)
