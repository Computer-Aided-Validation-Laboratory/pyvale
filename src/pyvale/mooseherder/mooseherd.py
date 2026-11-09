# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Parameter sweeps of isolated mesh/solve chains."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .directorymanager import DirectoryManager, EDirMode
from .simrunner import InputFiles, InputValues, RunResult, SimRunner


@dataclass(frozen=True, slots=True)
class SimStep:
    """Describe one named runner and its input tree within a simulation chain.

    Parameters
    ----------
    name_subdir : str
        Unique snake_case step name starting with a lowercase letter and containing
        lowercase letters, digits and underscores. Used as the case-parameter key and step
        subdirectory name.
    runner : SimRunner
        Runner to execute this step, such as GmshRunner or MooseRunner.
    input_files : Path | list[Path] | tuple[Path, ...]
        Entry input files accepted by the runner. Sequence order is significant
        for multiple MOOSE decks.
    input_root : Path | None, optional
        Input tree to snapshot. None uses a single entry file's parent.
    mesh_from : str | None, optional
        Name of an earlier step whose output is supplied as the mesh input.
        Defaults to None for independent/generated-mesh steps.
    mesh_parameter : str, optional
        Native MOOSE parameter receiving that output path. Defaults to "Mesh/file";
        use the deck's root mesh variable when replacing the downstream path would
        leave that variable unused. Case parameters cannot override this link.
    """

    name_subdir: str
    runner: SimRunner
    input_files: InputFiles
    input_root: Path | None = None
    mesh_from: str | None = None
    mesh_parameter: str = "Mesh/file"


class MooseHerd:
    """Execute named simulation chains sequentially or concurrently.

    Notes
    -----
    Each case maps step names to parameter dictionaries. Threads orchestrate
    isolated subprocesses, sharing no Gmsh state; notebook callers need no process
    main guard. ARCHIVE retains each batch under sweep_NNNN/run_NNNN/<step>.
    A REUSE DirectoryManager instead replaces stable reuse-ID directories.
    """

    def __init__(
        self,
        steps: list[SimStep],
        *,
        output_dir: Path | None = None,
        dir_manager: DirectoryManager | None = None,
        parallel_runners: int = 1,
    ) -> None:
        """Configure an ordered simulation chain and its directory policy.

        Parameters
        ----------
        steps : list[SimStep]
            Nonempty ordered chain with unique names. Mesh dependencies must refer
            to earlier steps.
        output_dir : Path | None, optional
            Archive output root. Supply either this or dir_manager.
        dir_manager : DirectoryManager | None, optional
            Explicit archive/reuse policy. None creates an archive manager from
            output_dir. Input trees must not overlap directories subject to replacement.
        parallel_runners : int, optional
            Positive maximum number of concurrent case chains. Defaults to 1.

        Raises
        ------
        ValueError
            The chain, dependencies, parallelism, input trees or directory arguments
            are invalid or ambiguous.
        """
        if not steps:
            raise ValueError("At least one simulation step is required.")

        if type(parallel_runners) is not int or parallel_runners < 1:
            raise ValueError("parallel_runners must be a positive integer.")

        names = set()
        for step in steps:
            if (
                not re.fullmatch(r"[a-z][a-z0-9_]*", step.name_subdir)
                or step.name_subdir in names
            ):
                raise ValueError(f"Invalid or duplicate step name: {step.name_subdir}")
            if step.mesh_from is not None and step.mesh_from not in names:
                raise ValueError("mesh_from must name an earlier step.")
            names.add(step.name_subdir)

        if dir_manager is not None and output_dir is not None:
            raise ValueError("Specify output_dir or dir_manager, not both.")

        if dir_manager is None:
            if output_dir is None:
                raise ValueError("Specify output_dir or dir_manager.")

            dir_manager = DirectoryManager(output_dir)

        for step in steps:
            paths = (
                (step.input_files,)
                if isinstance(step.input_files, Path)
                else tuple(step.input_files)
            )

            if not paths:
                raise ValueError("Each step requires at least one input file.")

            root = (step.input_root or paths[0].parent).resolve()

            if dir_manager.output_dir.is_relative_to(root) or (
                dir_manager.mode is EDirMode.REUSE
                and root.is_relative_to(dir_manager.output_dir)
            ):
                raise ValueError(
                    "Input trees must be outside managed output directories."
                )

        self.dir_manager = dir_manager
        self.steps = tuple(steps)
        self.parallel_runners = parallel_runners

    def _run_case(
        self,
        item: tuple[Path, dict[str, dict[str, InputValues]]],
    ) -> list[RunResult]:
        """Run one case chain and publish a reuse completion manifest when appropriate.

        Parameters
        ----------
        item : tuple[Path, dict[str, dict[str, InputValues]]]
            Reserved case directory and parameter mappings keyed by step name.

        Returns
        -------
        list[RunResult]
            Successful results in chain order. A failed step raises and prevents
            later steps in this chain from starting.
        """
        directory, case_params = item
        outputs: dict[str, RunResult] = {}

        for step in self.steps:
            input_params = dict(case_params.get(step.name_subdir, {}))
            if step.mesh_from is not None:
                if step.mesh_parameter in input_params:
                    raise ValueError(f"The chain manages {step.mesh_parameter}.")
                input_params[step.mesh_parameter] = str(
                    outputs[step.mesh_from].output_file
                )

            outputs[step.name_subdir] = step.runner.run(
                step.input_files,
                output_dir=directory / step.name_subdir,
                input_params=input_params,
                input_root=step.input_root,
            )

        results = list(outputs.values())
        if self.dir_manager.mode is EDirMode.REUSE:
            manifest = {
                "steps": [s.name_subdir for s in self.steps],
                "case_input_params": [case_params],
                "reuse_ids": [directory.name],
                "outputs": [
                    [r.output_file.relative_to(directory).as_posix() for r in results]
                ],
            }
            temporary = directory / ".evaluation.json.tmp"
            temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary.replace(directory / "evaluation.json")

        return results

    def run(
        self,
        case_input_params: list[dict[str, dict[str, InputValues]]],
        *,
        reuse_ids: list[str] | None = None,
    ) -> list[list[RunResult]]:
        """Execute a batch of cases and return results in their submitted order.

        Parameters
        ----------
        case_input_params : list[dict[str, dict[str, InputValues]]]
            Nonempty case list. Each dictionary maps configured step names to native
            parameter dictionaries; omitted steps use their model defaults.
        reuse_ids : list[str] | None, optional
            Stable reuse IDs in case order. Required for REUSE mode and rejected
            in ARCHIVE mode. IDs do not depend on worker assignment.

        Returns
        -------
        list[list[RunResult]]
            Results indexed first by submitted case, then by simulation step.

        Raises
        ------
        ValueError
            Cases are empty, step names/IDs are invalid, or a managed mesh parameter
            is also supplied by a case.
        RuntimeError
            An ID is reserved, a backend fails, or a successful output is missing.

        Notes
        -----
        ARCHIVE preserves completed and incomplete runs; only complete batches receive
        sweep.json. REUSE removes old outputs before launch and atomically publishes an
        evaluation.json per successful chain. Failed chains have no completion marker.
        Read reusable output arrays before submitting the same IDs again.
        """
        if not case_input_params:
            raise ValueError("At least one sweep case is required.")

        names = {s.name_subdir for s in self.steps}

        for case_params in case_input_params:
            if set(case_params) - names:
                raise ValueError(f"Unknown step names: {set(case_params) - names}")

            for step in self.steps:
                if (
                    step.mesh_from is not None
                    and step.mesh_parameter in case_params.get(step.name_subdir, {})
                ):
                    raise ValueError(f"The chain manages {step.mesh_parameter}.")

        with self.dir_manager.reserve(len(case_input_params), reuse_ids) as allocation:
            sweep_dir, directories = allocation
            jobs = list(zip(directories, case_input_params))

            if self.parallel_runners == 1:
                results = [self._run_case(job) for job in jobs]
            else:
                with ThreadPoolExecutor(max_workers=self.parallel_runners) as executor:
                    results = list(executor.map(self._run_case, jobs))

            if sweep_dir is not None:
                manifest = {
                    "steps": [s.name_subdir for s in self.steps],
                    "case_input_params": case_input_params,
                    "outputs": [
                        [r.output_file.relative_to(sweep_dir).as_posix() for r in row]
                        for row in results
                    ],
                }
                temporary = sweep_dir / ".sweep.json.tmp"
                temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                temporary.replace(sweep_dir / "sweep.json")

            return results
