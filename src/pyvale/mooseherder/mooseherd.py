"""Parameter sweeps of isolated mesh/solve chains."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

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
    """

    def __init__(
        self,
        steps: list[SimulationStep],
        *,
        output_dir: Path,
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
        self.steps = tuple(steps)
        self.output_dir = output_dir.resolve()
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
        return list(outputs.values())

    def run(
        self,
        cases: list[dict[str, dict[str, MooseValue]]],
    ) -> list[list[RunResult]]:
        """Execute cases in input order and write a portable result manifest.

        An incomplete sweep has no sweep.json; subprocess errors propagate and
        prevent later steps in the failing chain from executing. Existing
        sweeps are preserved, including incomplete runs, on subsequent calls.
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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        number = 1
        while True:
            sweep_dir = self.output_dir / f"sweep-{number:04d}"
            try:
                sweep_dir.mkdir()
                break
            except FileExistsError:
                number += 1
        jobs = [(sweep_dir / f"run-{i:04d}", case) for i, case in enumerate(cases, 1)]
        if self.num_parallel == 1:
            results = [self._run_case(job) for job in jobs]
        else:
            with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
                results = list(executor.map(self._run_case, jobs))
        manifest = {
            "steps": [s.name for s in self.steps],
            "cases": cases,
            "outputs": [
                [str(r.output.relative_to(sweep_dir)) for r in row] for row in results
            ],
        }
        (sweep_dir / "sweep.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return results
