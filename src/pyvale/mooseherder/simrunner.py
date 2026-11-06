"""Typed run inputs, isolated input staging and subprocess execution."""

import json
import math
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

MooseScalar: TypeAlias = str | bool | int | float
MooseValue: TypeAlias = MooseScalar | list[MooseScalar] | tuple[MooseScalar, ...]
InputFiles: TypeAlias = Path | list[Path] | tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class RunResult:
    """One completed step, including its output, log and reproducible command."""

    output: Path
    directory: Path
    command: tuple[str, ...]
    log: Path


def parameter_args(parameters: dict[str, MooseValue]) -> list[str]:
    """Serialize MOOSE overrides as individual arguments, without shell quoting.

    Names are native MOOSE names or slash-separated block paths. Sequences
    represent MOOSE vectors. Empty vectors, nonfinite floats and nested values
    are rejected; MOOSE validates parameter existence and physical meaning.
    """
    args = []
    for name, value in parameters.items():
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)*", name
        ):
            raise ValueError(f"Invalid parameter name: {name!r}")
        values = value if isinstance(value, (list, tuple)) else [value]
        if not values:
            raise ValueError(f"Empty parameter vector: {name}")
        words = []
        for item in values:
            if isinstance(item, bool):
                words.append("true" if item else "false")
            elif isinstance(item, int):
                words.append(str(item))
            elif isinstance(item, float):
                if not math.isfinite(item):
                    raise ValueError(f"Nonfinite parameter: {name}")
                words.append(repr(item))
            elif isinstance(item, str) and not any(c in item for c in "\n\r\0"):
                words.append(item)
            else:
                raise TypeError(f"Unsupported value for {name}: {item!r}")
        args.append(f"{name}={' '.join(words)}")
    return args


def stage_inputs(
    inputs: InputFiles, directory: Path, input_root: Path | None
) -> tuple[Path, ...]:
    """Snapshot a declared input tree, preserving nested include/asset paths.

    A single entry defaults to its parent directory. Multiple entry decks must
    share an explicit input_root. Dependencies outside this root must be moved
    into it or referenced explicitly by absolute paths in the user's inputs.
    """
    paths = (inputs,) if isinstance(inputs, Path) else tuple(inputs)
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
    directory = directory.resolve()
    if directory.is_relative_to(root):
        raise ValueError("The output directory must be outside the input tree.")
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / "inputs"
    shutil.copytree(root, target)
    return tuple(target / p for p in relative)


def execute(
    command: list[str],
    directory: Path,
    parameters: dict[str, MooseValue],
    *,
    env: dict[str, str] | None = None,
    timeout: float = 180.0,
) -> Path:
    """Record a run and stream output to disk; raise on failure or timeout."""
    (directory / "parameters.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8"
    )
    (directory / "command.json").write_text(
        json.dumps(command, indent=2), encoding="utf-8"
    )
    log = directory / "run.log"
    with log.open("w", encoding="utf-8") as stream:
        try:
            subprocess.run(
                command,
                cwd=directory,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as err:
            raise RuntimeError(f"Simulation failed; see {log}") from err
    return log


class SimRunner(ABC):
    """A stateless simulation step with explicit inputs and an isolated output."""

    @abstractmethod
    def run(
        self,
        inputs: InputFiles | None = None,
        *,
        output_dir: Path,
        parameters: dict[str, MooseValue] | None = None,
        input_root: Path | None = None,
    ) -> RunResult:
        """Execute a step and return its actual output path."""


def next_run_path(path: Path) -> Path:
    """Choose an unused output path for rerunnable notebooks and examples.

    Return path itself when unused, otherwise append -2, -3, and so on. The
    runner still reserves the directory exclusively when starting the run;
    this helper never deletes or overwrites earlier results.
    """
    candidate = path
    number = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}-{number}")
        number += 1
    return candidate
