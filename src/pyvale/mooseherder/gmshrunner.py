"""Generate meshes from geometry files or user Python scripts in subprocesses."""

import json
import os
import sys
from pathlib import Path

from .availability import BackendAvailability, gmsh_availability
from .simrunner import (
    InputFiles,
    MooseValue,
    RunResult,
    SimRunner,
    execute,
    parameter_args,
    stage_inputs,
)


class GmshRunner(SimRunner):
    """Mesh .geo inputs or execute a supplied .py mesh script.

    Python scripts accept ``--config <JSON>`` containing ``parameters`` and
    ``output_path`` and must write that mesh. Geometry parameters use Gmsh's
    -setnumber/-setstring convention (use DefineConstant in the .geo file).
    Existing Mesh/Save commands are supported, but Exit commands must be
    removed: successful completion and the requested output are both checked.
    An explicit executable selects external Gmsh for .geo files only.
    """

    def __init__(
        self,
        *,
        geometry: Path | None = None,
        script: Path | None = None,
        executable: Path | None = None,
        dimension: int = 2,
        order: int | None = None,
        timeout: float = 180.0,
    ) -> None:
        if geometry is not None and script is not None:
            raise ValueError("Specify geometry or script, not both.")
        if dimension not in (1, 2, 3):
            raise ValueError("dimension must be 1, 2 or 3.")
        if order is not None and (type(order) is not int or order < 1):
            raise ValueError("order must be a positive integer.")
        if timeout <= 0:
            raise ValueError("timeout must be positive.")
        self.input_file = geometry if geometry is not None else script
        self.executable = executable.resolve() if executable is not None else None
        self.dimension = dimension
        self.order = order
        self.timeout = timeout

    def availability(self) -> BackendAvailability:
        """Check the selected Gmsh installation."""
        return gmsh_availability(self.executable)

    def run(
        self,
        inputs: InputFiles | None = None,
        *,
        output_dir: Path,
        parameters: dict[str, MooseValue] | None = None,
        input_root: Path | None = None,
    ) -> RunResult:
        source = inputs if inputs is not None else self.input_file
        if not isinstance(source, Path) or source.suffix not in (".geo", ".py"):
            raise ValueError("Specify one .geo geometry or .py mesh script.")
        if source.suffix == ".py" and self.executable is not None:
            raise ValueError("A custom Gmsh executable cannot execute Python scripts.")
        values = dict(parameters or {})
        parameter_args(values)
        if source.suffix == ".geo" and any(
            isinstance(value, (list, tuple)) for value in values.values()
        ):
            raise TypeError("Gmsh geometry parameters must be scalar.")
        availability = self.availability()
        if not availability.available:
            raise RuntimeError(availability.reason)
        directory = output_dir.resolve()
        staged = stage_inputs(source, directory, input_root)[0]
        output = directory / "mesh.msh"
        config = directory / "mesh-config.json"
        config.write_text(
            json.dumps(
                {
                    "input_path": str(staged),
                    "output_path": str(output),
                    "parameters": values,
                    "dimension": self.dimension,
                    "order": self.order,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if staged.suffix == ".py":
            command = [sys.executable, str(staged), "--config", str(config)]
        elif self.executable is not None:
            command = [
                str(self.executable),
                str(staged),
                f"-{self.dimension}",
                "-o",
                str(output),
                "-format",
                "msh2",
            ]
            if self.order is not None:
                command.extend(("-order", str(self.order)))
            for name, value in values.items():
                if isinstance(value, (int, float)):
                    command.extend(
                        (
                            "-setnumber",
                            name,
                            str(int(value) if isinstance(value, bool) else value),
                        )
                    )
                elif isinstance(value, str):
                    command.extend(("-setstring", name, value))
                else:
                    raise TypeError("Gmsh geometry parameters must be scalar.")
        else:
            # An absolute worker path works for source trees and installed wheels,
            # and does not import the whole PyVale package in the child process.
            worker = Path(__file__).with_name("_gmshworker.py")
            command = [sys.executable, str(worker), str(config)]
        log = execute(
            command, directory, values, env=os.environ.copy(), timeout=self.timeout
        )
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"Gmsh did not produce {output}; see {log}")
        return RunResult(output, directory, tuple(command), log)
