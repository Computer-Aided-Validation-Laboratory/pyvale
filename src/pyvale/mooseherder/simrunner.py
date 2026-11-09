# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Typed run inputs, isolated input staging and subprocess execution."""

import json
import math
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

InputScalar: TypeAlias = str | bool | int | float
InputValues: TypeAlias = InputScalar | list[InputScalar] | tuple[InputScalar, ...]
InputFiles: TypeAlias = Path | list[Path] | tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class RunnerBackend:
    """Resolved resource used by a runner after successful backend checks.

    Parameters
    ----------
    backend_file : Path
        Absolute executable path for MOOSE or custom Gmsh, or the gmsh.py module
        path for Python Gmsh. Unusable backends raise instead of returning a record.
    """

    backend_file: Path


def _check_backend_exe(exe_file: Path) -> Path:
    """Resolve an executable path and check file existence and OS permissions.

    Parameters
    ----------
    exe_file : Path
        Backend executable to inspect without running it.

    Returns
    -------
    Path
        Absolute executable path.

    Raises
    ------
    FileNotFoundError
        The path does not identify a file.
    PermissionError
        The current OS user cannot execute the file.
    """
    exe_file = exe_file.resolve()
    if not exe_file.is_file():
        raise FileNotFoundError(f"Backend executable not found: {exe_file}")
    if not os.access(exe_file, os.X_OK):
        raise PermissionError(f"Backend file is not executable: {exe_file}")
    return exe_file


def _check_output_file_name(output_file_name: str) -> None:
    """Check that a requested output name stays inside its working directory.

    Parameters
    ----------
    output_file_name : str
        Nonempty filename or file base, without directory components.

    Raises
    ------
    ValueError
        The name is empty, a path component, or contains a separator/control code.
    TypeError
        The name is not a string.
    """
    if not isinstance(output_file_name, str):
        raise TypeError("Output filenames must be strings.")
    if (
        not output_file_name
        or output_file_name in {".", ".."}
        or any(c in output_file_name for c in "/\\\0\r\n")
    ):
        raise ValueError("Output filenames must be basenames without directories.")


@dataclass(frozen=True, slots=True)
class RunResult:
    """Output paths and invocation metadata for one successful simulation step.

    Parameters
    ----------
    output_file : Path
        Absolute primary output path, such as results.e or mesh.msh.
    output_dir : Path
        Absolute working directory containing staged input files and run metadata.
    command : tuple[str, ...]
        Exact argument vector supplied to the subprocess, without a shell.
    log_file : Path
        Combined standard-output and standard-error log for this step.

    Notes
    -----
    The record is immutable, but its paths refer to files on disk. Read results
    before reusing an evaluation ID: REUSE mode replaces that ID's old files.
    """

    output_file: Path
    output_dir: Path
    command: tuple[str, ...]
    log_file: Path


def conv_args_dict_to_list(input_params: dict[str, InputValues]) -> list[str]:
    """Serialise native MOOSE parameter overrides into subprocess arguments.

    Parameters
    ----------
    input_params : dict[str, InputValues]
        Root variable names or slash-separated block paths mapped to strings,
        booleans, integers, finite floats, or flat sequences of those scalars.
        Sequences represent MOOSE vectors; values use the input deck's units.

    Returns
    -------
    list[str]
        One name=value argument per parameter, in dictionary insertion order.
        Boolean values use true/false and vector entries are space-separated.

    Raises
    ------
    ValueError
        A name is malformed, a vector is empty, or a float is nonfinite.
    TypeError
        A value is unsupported, nested, or contains a prohibited control character.

    Notes
    -----
    Arguments are passed with shell=False. Callers do not add shell quoting.
    MOOSE validates whether a parameter exists and is physically meaningful.
    """
    args: list[str] = []
    for name, value in input_params.items():
        validate_parameter_name(name)
        values = value if isinstance(value, (list, tuple)) else [value]
        if not values:
            raise ValueError(f"Empty parameter vector: {name}")
        words = [format_scalar(item) for item in values]
        args.append(f"{name}={' '.join(words)}")
    return args


def validate_parameter_name(name: str) -> None:
    """Validate that a parameter name can form one CLI assignment.

    Parameters
    ----------
    name : str
        Native parameter name; the simulation backend validates its grammar.

    Raises
    ------
    TypeError
        The name is not a string.
    ValueError
        The name is empty, resembles an option or contains whitespace, NUL or =.
    """
    if not isinstance(name, str):
        raise TypeError("Parameter names must be strings.")
    if not name or name.startswith("-") or any(c.isspace() or c in "=\0" for c in name):
        raise ValueError(f"Invalid parameter name: {name!r}")


def format_scalar(value: InputScalar) -> str:
    """Convert a supported input scalar to its native CLI representation.

    Parameters
    ----------
    value : InputScalar
        String, boolean, integer or finite floating-point value.

    Returns
    -------
    str
        Scalar text, with booleans represented by true/false.

    Raises
    ------
    TypeError
        The value is unsupported or contains a prohibited control character.
    ValueError
        A floating-point value is nonfinite.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if not isinstance(value, (str, int, float)):
        raise TypeError(f"Unsupported parameter value: {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite parameter value: {value}")
    text = str(value)
    if any(c in text for c in "\r\n\0"):
        raise TypeError("Parameter values cannot contain line breaks or NUL.")
    return text


def stage_input_files(
    input_files: InputFiles, work_dir: Path, input_root: Path | None
) -> tuple[Path, ...]:
    """Copy an input tree while retaining relative include and asset paths.

    Parameters
    ----------
    input_files : Path | list[Path] | tuple[Path, ...]
        One entry file or an ordered sequence of entry decks.
    work_dir : Path
        New working directory, outside the declared input tree.
    input_root : Path | None
        Tree to snapshot. None uses a single entry file's parent; multiple entry
        decks require an explicit root containing every entry.

    Returns
    -------
    tuple[Path, ...]
        Absolute staged entry paths, preserving their original order and relative
        locations below work_dir/input_files.

    Raises
    ------
    FileNotFoundError
        An entry file does not exist.
    FileExistsError
        The working directory already exists.
    ValueError
        Entries are empty, a required root is absent, an entry is outside the root,
        or the output directory is inside the input tree.

    Notes
    -----
    Files outside the declared tree are not discovered automatically. Move relative
    assets into the tree or explicitly reference external absolute paths in decks.
    """
    paths = (input_files,) if isinstance(input_files, Path) else tuple(input_files)
    if not paths:
        raise ValueError("At least one input file is required.")

    paths = tuple(p.resolve() for p in paths)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    if input_root is None:
        if len(paths) > 1:
            raise ValueError("Specify input_root for multiple entry decks.")
        input_root = paths[0].parent

    root = input_root.resolve()
    relative = tuple(p.relative_to(root) for p in paths)
    work_dir = work_dir.resolve()
    if work_dir.is_relative_to(root):
        raise ValueError("The output directory must be outside the input tree.")

    work_dir.mkdir(parents=True, exist_ok=False)
    target = work_dir / "input_files"
    shutil.copytree(root, target)

    return tuple(target / p for p in relative)


def execute(
    command: list[str],
    work_dir: Path,
    input_params: dict[str, InputValues],
    *,
    env_vars: dict[str, str] | None = None,
    timeout: float = 3600.0,
) -> Path:
    """Record a subprocess invocation and stream its output to the run log.

    Parameters
    ----------
    command : list[str]
        Application argument vector. Execution uses shell=False.
    work_dir : Path
        Existing working directory containing the staged input files.
    input_params : dict[str, InputValues]
        Native parameter values to record in input_params.json.
    env_vars : dict[str, str] | None, optional
        Complete mapping of OS environment variables for the subprocess.
        A supplied mapping replaces the calling Python process's OS environment
        variables for this subprocess. None inherits the calling Python process's
        OS environment variables (os.environ). This does not select or create
        a Python virtual environment.
    timeout : float, optional
        Maximum process runtime in seconds. Defaults to 3600 (1 hour).

    Returns
    -------
    Path
        Combined stdout/stderr log at work_dir/run.log. command.json and
        input_params.json are also written in the working directory.

    Raises
    ------
    RuntimeError
        The process returns a nonzero status or exceeds the timeout. The exception
        identifies the retained log and chains the subprocess failure.
    OSError
        Metadata/log files cannot be written or the process cannot be started.
    """
    (work_dir / "input_params.json").write_text(
        json.dumps(input_params, indent=2), encoding="utf-8"
    )
    (work_dir / "command.json").write_text(
        json.dumps(command, indent=2), encoding="utf-8"
    )

    log = work_dir / "run.log"
    with log.open("w", encoding="utf-8") as stream:
        try:
            subprocess.run(
                command,
                cwd=work_dir,
                env=env_vars,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as err:
            raise RuntimeError(f"Simulation failed; see {log}") from err

    return log


class SimRunner(ABC):
    """Interface for a simulation step with explicit input files and output ownership.

    Notes
    -----
    Implementations stage an input tree, execute in an isolated working directory,
    raise on execution failure, and return a RunResult only for successful output.
    Parameter values retain the selected input model's units and conventions.
    """

    @abstractmethod
    def run(
        self,
        input_files: InputFiles,
        *,
        output_dir: Path,
        input_params: dict[str, InputValues] | None = None,
        input_root: Path | None = None,
    ) -> RunResult:
        """Execute one step and return its actual output paths.

        Parameters
        ----------
        input_files : Path | list[Path] | tuple[Path, ...]
            Entry input file or ordered entry decks, as supported by the concrete runner.
        output_dir : Path
            New working directory outside the input tree.
        input_params : dict[str, InputValues] | None, optional
            Native input parameter values. None uses the model defaults.
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
        """


def get_next_run_path(path: Path) -> Path:
    """Choose an unused output path for a rerunnable example or notebook.

    Parameters
    ----------
    path : Path
        Preferred output path.

    Returns
    -------
    Path
        The preferred path when unused, otherwise a sibling suffixed _1, _2, etc.

    Notes
    -----
    This lookup never removes existing results and does not reserve the path.
    The runner reserves its working directory exclusively when execution begins.
    """
    candidate = path
    number = 1

    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{number}")
        number += 1

    return candidate
