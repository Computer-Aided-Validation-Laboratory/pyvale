"""Parameter sweeps of isolated mesh/solve chains."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .directorymanager import DirectoryManager, EDirectoryMode
from .simrunner import InputFiles, MooseValue, RunResult, SimRunner


@dataclass(frozen=True, slots=True)
class SimulationStep:
    """One named step and its input tree.

    mesh_from names an earlier step whose output is supplied to the MOOSE
    mesh_parameter. It cannot be overridden by case parameters. By default
    MOOSE FileMesh uses Mesh/file; generated-mesh decks need no mesh_from.
    """

    name: str
    runner: SimRunner
    inputs: InputFiles
    input_root: Path | None = None
    mesh_from: str | None = None
    mesh_parameter: str = "Mesh/file"


class MooseHerd:
    """Run chains sequentially or concurrently with immutable per-case inputs.

    Each case maps step names to parameter dictionaries. Parallel workers are
    threads orchestrating isolated subprocesses; no Gmsh state is shared and
    notebook users do not need a multiprocessing main guard. Results and a
    sweep.json manifest are written under sweep-NNNN/run-NNNN/<step>.
    A REUSE DirectoryManager instead replaces stable evaluation-ID directories.
    """

    def __init__(
        self,
        steps: list[SimulationStep],
        *,
        output_dir: Path | None = None,
        directory_manager: DirectoryManager | None = None,
        num_parallel: int = 1,
    ) -> None:
        if not steps:
            raise ValueError("At least one simulation step is required.")
        if type(num_parallel) is not int or num_parallel < 1:
            raise ValueError("num_parallel must be a positive integer.")
        names = set()
        for step in steps:
            if (
                not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", step.name)
                or step.name in names
            ):
                raise ValueError(f"Invalid or duplicate step name: {step.name}")
            if step.mesh_from is not None and step.mesh_from not in names:
                raise ValueError("mesh_from must name an earlier step.")
            names.add(step.name)
        if directory_manager is not None and output_dir is not None:
            raise ValueError("Specify output_dir or directory_manager, not both.")
        if directory_manager is None:
            if output_dir is None:
                raise ValueError("Specify output_dir or directory_manager.")
            directory_manager = DirectoryManager(output_dir)
        for step in steps:
            paths = (
                (step.inputs,) if isinstance(step.inputs, Path) else tuple(step.inputs)
            )
            if not paths:
                raise ValueError("Each step requires at least one input file.")
            root = (step.input_root or paths[0].parent).resolve()
            if directory_manager.output_dir.is_relative_to(root) or (
                directory_manager.mode is EDirectoryMode.REUSE
                and root.is_relative_to(directory_manager.output_dir)
            ):
                raise ValueError(
                    "Input trees must be outside managed output directories."
                )
        self.directory_manager = directory_manager
        self.steps = tuple(steps)
        self.num_parallel = num_parallel

    def _run_case(
        self,
        item: tuple[Path, dict[str, dict[str, MooseValue]]],
    ) -> list[RunResult]:
        directory, case = item
        outputs: dict[str, RunResult] = {}
        for step in self.steps:
            parameters = dict(case.get(step.name, {}))
            if step.mesh_from is not None:
                if step.mesh_parameter in parameters:
                    raise ValueError(f"The chain manages {step.mesh_parameter}.")
                parameters[step.mesh_parameter] = str(outputs[step.mesh_from].output)
            outputs[step.name] = step.runner.run(
                step.inputs,
                output_dir=directory / step.name,
                parameters=parameters,
                input_root=step.input_root,
            )
        results = list(outputs.values())
        if self.directory_manager.mode is EDirectoryMode.REUSE:
            manifest = {
                "steps": [s.name for s in self.steps],
                "cases": [case],
                "evaluation_ids": [directory.name],
                "outputs": [[str(r.output.relative_to(directory)) for r in results]],
            }
            temporary = directory / ".evaluation.json.tmp"
            temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary.replace(directory / "evaluation.json")
        return results

    def run(
        self,
        cases: list[dict[str, dict[str, MooseValue]]],
        *,
        evaluation_ids: list[str] | None = None,
    ) -> list[list[RunResult]]:
        """Execute cases in input order and write a portable result manifest.

        ARCHIVE preserves each batch, including incomplete runs. REUSE requires
        stable evaluation_ids and replaces their previous outputs. Errors stop
        the failing chain; no completion manifest is written for that chain.
        Read reusable results before submitting the same IDs again.
        """
        if not cases:
            raise ValueError("At least one sweep case is required.")
        names = {s.name for s in self.steps}
        for case in cases:
            if set(case) - names:
                raise ValueError(f"Unknown step names: {set(case) - names}")
            for step in self.steps:
                if step.mesh_from is not None and step.mesh_parameter in case.get(
                    step.name, {}
                ):
                    raise ValueError(f"The chain manages {step.mesh_parameter}.")
        with self.directory_manager.reserve(len(cases), evaluation_ids) as allocation:
            sweep_dir, directories = allocation
            jobs = list(zip(directories, cases))
            if self.num_parallel == 1:
                results = [self._run_case(job) for job in jobs]
            else:
                with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
                    results = list(executor.map(self._run_case, jobs))
            if sweep_dir is not None:
                manifest = {
                    "steps": [s.name for s in self.steps],
                    "cases": cases,
                    "evaluation_ids": evaluation_ids,
                    "outputs": [
                        [str(r.output.relative_to(sweep_dir)) for r in row]
                        for row in results
                    ],
                }
                temporary = sweep_dir / ".sweep.json.tmp"
                temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                temporary.replace(sweep_dir / "sweep.json")
            return results
