"""Execute MOOSE decks with packaged Rabbit or an explicitly selected app."""

import os
import shutil
import sys
from pathlib import Path

from .availability import BackendAvailability, moose_availability
from .simrunner import (
    InputFiles,
    MooseValue,
    RunResult,
    SimRunner,
    execute,
    parameter_args,
    stage_inputs,
)


class MooseRunner(SimRunner):
    """Run ordered MOOSE decks with native CLI overrides.

    Rabbit defaults to serial/SMP. MPI requires an MPI-capable Rabbit build or
    custom executable and a launcher. ``mpi`` controls launch policy; select
    the matching solver entry deck explicitly. Each run stages input_root and
    writes results.e, run.log, command.json and parameters.json. Existing run
    directories are rejected, so stale results cannot masquerade as new ones.
    """

    def __init__(
        self,
        *,
        executable: Path | None = None,
        mpi: bool = False,
        n_tasks: int = 1,
        n_threads: int = 1,
        launcher: tuple[str, ...] = ("mpirun", "-np"),
        env: dict[str, str] | None = None,
        timeout: float = 180.0,
        output_parameter: str = "Outputs/file_base",
    ) -> None:
        if type(n_tasks) is not int or n_tasks < 1:
            raise ValueError("n_tasks must be a positive integer.")
        if type(n_threads) is not int or n_threads < 1:
            raise ValueError("n_threads must be a positive integer.")
        if not mpi and n_tasks != 1:
            raise ValueError("Multiple MPI tasks require mpi=True.")
        if timeout <= 0:
            raise ValueError("timeout must be positive.")
        self.executable = executable.resolve() if executable is not None else None
        self.mpi = mpi
        self.n_tasks = n_tasks
        self.n_threads = n_threads
        self.launcher = launcher
        self.env = dict(env or {})
        self.timeout = timeout
        parameter_args({output_parameter: "results"})
        self.output_parameter = output_parameter

    def availability(self) -> BackendAvailability:
        """Inspect the selected executable without starting a simulation."""
        return moose_availability(self.executable, mpi=self.mpi)

    def run(
        self,
        inputs: InputFiles | None = None,
        *,
        output_dir: Path,
        parameters: dict[str, MooseValue] | None = None,
        input_root: Path | None = None,
    ) -> RunResult:
        if inputs is None:
            raise ValueError("Specify a MOOSE input deck or ordered sequence.")
        values = dict(parameters or {})
        # The runner owns the primary output contract; do not silently replace
        # a user's request for a different output location.
        if {
            "Outputs/file_base",
            "Outputs/exodus",
            self.output_parameter,
        } & values.keys():
            raise ValueError("The runner manages Outputs/file_base and Outputs/exodus.")
        overrides = parameter_args(values)
        availability = self.availability()
        if not availability.available:
            raise RuntimeError(availability.reason)
        command = []
        if self.mpi:
            if not self.launcher or shutil.which(self.launcher[0]) is None:
                raise RuntimeError("MPI launcher is unavailable; configure launcher.")
            command.extend((*self.launcher, str(self.n_tasks)))
        if self.executable is None:
            # The public CLI sets Rabbit's library search environment and checks
            # its build variant. Use this interpreter, never PATH's Python.
            command.extend(
                (sys.executable, "-c", "from rabbit.cli import main; main()")
            )
        else:
            command.append(str(self.executable))
        directory = output_dir.resolve()
        staged = stage_inputs(inputs, directory, input_root)
        command.extend(("-i", *(str(p.relative_to(directory)) for p in staged)))
        command.extend((f"--n-threads={self.n_threads}", *overrides))
        command.extend((f"{self.output_parameter}=results", "Outputs/exodus=true"))
        environment = os.environ.copy()
        environment.update(self.env)
        log = execute(command, directory, values, env=environment, timeout=self.timeout)
        output = directory / "results.e"
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"MOOSE did not produce {output}; see {log}")
        return RunResult(output, directory, tuple(command), log)
