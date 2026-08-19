"""Local-array and Slurm workflow executors."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
import json
import multiprocessing as mp
from pathlib import Path
import subprocess

from .case import WorkflowCase
from .config import EFailurePolicy, EWorkflowStorage, WorkflowConfig
from .pipeline import IWorkflow
from .result import CaseResult
from .runner import _run_case


@dataclass(frozen=True, slots=True)
class SlurmConfig:
    """Slurm resource and submission settings for a workflow job array."""

    partition: str
    account: str | None = None
    qos: str | None = None
    wall_time: str = "01:00:00"
    cpus_per_task: int = 1
    memory: str | None = None
    array_concurrency: int | None = None
    modules: tuple[str, ...] = ()
    setup_commands: tuple[str, ...] = ()
    python_executable: str = "python"
    submit: bool = True

    def __post_init__(self) -> None:
        """Validate scheduler resource requests before script generation."""
        if not self.partition:
            raise ValueError("Slurm partition must not be empty.")
        if self.cpus_per_task <= 0:
            raise ValueError("Slurm cpus_per_task must be positive.")
        if self.array_concurrency is not None and self.array_concurrency <= 0:
            raise ValueError("Slurm array_concurrency must be positive or None.")


@dataclass(frozen=True, slots=True)
class SlurmSubmission:
    """Prepared Slurm workflow run and its optional submitted job identifier."""

    run_dir: Path
    script_path: Path
    case_count: int
    job_id: str | None = None


class IWorkflowExecutor(ABC):
    """Execute persisted workflow cases through a local or remote scheduler."""

    @abstractmethod
    def run(
        self,
        workflow_factory: str,
        cases: Iterable[WorkflowCase],
        config: WorkflowConfig,
    ) -> tuple[CaseResult, ...]:
        """Run cases constructed by an importable zero-argument factory."""


class LocalArrayExecutor(IWorkflowExecutor):
    """Test Slurm-style persisted worker execution on the local machine."""

    def __init__(self, workers: int = 1) -> None:
        """Store the number of local process workers."""
        if workers <= 0:
            raise ValueError("workers must be positive.")
        self.workers = workers

    def run(
        self,
        workflow_factory: str,
        cases: Iterable[WorkflowCase],
        config: WorkflowConfig,
    ) -> tuple[CaseResult, ...]:
        """Prepare a run then execute its persisted case files locally."""
        submission = SlurmExecutor(
            SlurmConfig(partition="local", submit=False),
        ).prepare(workflow_factory, cases, config)
        case_indices = list(range(submission.case_count))
        if self.workers == 1:
            results = [_run_worker_array_index(submission.run_dir, index)
                       for index in case_indices]
        else:
            context = mp.get_context("spawn")
            with context.Pool(self.workers) as pool:
                results = pool.starmap(
                    _run_worker_array_index,
                    [(submission.run_dir, index) for index in case_indices],
                )
        return tuple(sorted(results, key=lambda result: result.case.index))


class SlurmExecutor:
    """Prepare, optionally submit, and gather Slurm job-array workflows."""

    def __init__(self, slurm_config: SlurmConfig) -> None:
        """Store Slurm resource controls."""
        self.slurm_config = slurm_config

    def prepare(
        self,
        workflow_factory: str,
        cases: Iterable[WorkflowCase],
        config: WorkflowConfig,
    ) -> SlurmSubmission:
        """Persist cases and generate an ``sbatch`` script without submitting."""
        workflow = _build_workflow(workflow_factory)
        prepared_cases = tuple(cases)
        if config.max_cases is not None and len(prepared_cases) > config.max_cases:
            raise ValueError("Workflow case count exceeds WorkflowConfig.max_cases.")
        if self.slurm_config.cpus_per_task != config.threads_per_case:
            raise ValueError("Slurm cpus_per_task must equal threads_per_case.")
        for case in prepared_cases:
            workflow.verify_case(case)
        run_dir = config.output_dir
        cases_dir = run_dir / "cases"
        slurm_dir = run_dir / "slurm"
        cases_dir.mkdir(parents=True, exist_ok=True)
        slurm_dir.mkdir(parents=True, exist_ok=True)
        for case in prepared_cases:
            case_dir = cases_dir / f"{case.index:06d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            _write_case_manifest(case_dir, case)
        _write_workflow_manifest(run_dir, workflow_factory, config)
        indices_path = slurm_dir / "case-indices.txt"
        indices_path.write_text(
            "\n".join(str(case.index) for case in prepared_cases) + "\n",
            encoding="utf-8",
        )
        script_path = slurm_dir / "workflow.sbatch"
        script_path.write_text(
            _slurm_script(run_dir, self.slurm_config, len(prepared_cases)),
            encoding="utf-8",
        )
        return SlurmSubmission(run_dir, script_path, len(prepared_cases))

    def submit(
        self,
        workflow_factory: str,
        cases: Iterable[WorkflowCase],
        config: WorkflowConfig,
    ) -> SlurmSubmission:
        """Prepare and optionally submit a Slurm array through ``sbatch``."""
        submission = self.prepare(workflow_factory, cases, config)
        if not self.slurm_config.submit:
            return submission
        completed = subprocess.run(
            ("sbatch", "--parsable", str(submission.script_path)),
            check=True,
            capture_output=True,
            text=True,
        )
        job_id = completed.stdout.strip().split(";", maxsplit=1)[0]
        metadata = {"job_id": job_id, "script": str(submission.script_path)}
        (submission.run_dir / "slurm" / "submission.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8",
        )
        return SlurmSubmission(
            submission.run_dir, submission.script_path, submission.case_count,
            job_id,
        )


def _run_worker_array_index(run_dir: Path, array_index: int) -> CaseResult:
    """Run one persisted array entry using the same function as Slurm workers."""
    indices = (Path(run_dir) / "slurm" / "case-indices.txt").read_text().splitlines()
    return _run_persisted_case(Path(run_dir), int(indices[array_index]))


def _run_persisted_case(run_dir: Path, case_index: int) -> CaseResult:
    """Reconstruct a workflow and execute one persisted case manifest."""
    manifest = json.loads((run_dir / "workflow.json").read_text(encoding="utf-8"))
    config = _config_from_manifest(manifest["config"])
    workflow = _build_workflow(manifest["workflow_factory"])
    case_path = run_dir / "cases" / f"{case_index:06d}" / "case.json"
    case_data = json.loads(case_path.read_text(encoding="utf-8"))
    case = WorkflowCase(case_data["index"], case_data["values"], case_data["seed"])
    workflow.verify_case(case)
    return _run_case(workflow, config, case)


def _build_workflow(factory_path: str) -> IWorkflow:
    """Import and call a ``module:function`` workflow factory."""
    module_name, separator, function_name = factory_path.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("workflow_factory must use the 'module:function' form.")
    workflow = getattr(import_module(module_name), function_name)()
    if not isinstance(workflow, IWorkflow):
        raise TypeError("Workflow factory must return IWorkflow.")
    return workflow


def _write_case_manifest(case_dir: Path, case: WorkflowCase) -> None:
    """Persist the full deterministic case definition required by a worker."""
    data = {"index": case.index, "seed": case.seed, "values": dict(case.values)}
    (case_dir / "case.json").write_text(
        json.dumps(data, indent=2, default=_json_default), encoding="utf-8",
    )
    (case_dir / "parameters.json").write_text(
        json.dumps(dict(case.values), indent=2, default=_json_default),
        encoding="utf-8",
    )


def _write_workflow_manifest(
    run_dir: Path,
    workflow_factory: str,
    config: WorkflowConfig,
) -> None:
    """Persist the importable factory and portable execution configuration."""
    data = {
        "workflow_factory": workflow_factory,
        "config": {
            "output_dir": str(config.output_dir),
            "workers": config.workers,
            "threads_per_case": config.threads_per_case,
            "retain_artifacts": config.retain_artifacts,
            "failure_policy": config.failure_policy.value,
            "storage": config.storage.value,
            "max_in_flight": config.max_in_flight,
            "max_cases": config.max_cases,
        },
    }
    (run_dir / "workflow.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def _config_from_manifest(data: dict[str, object]) -> WorkflowConfig:
    """Reconstruct ``WorkflowConfig`` from its portable JSON representation."""
    return WorkflowConfig(
        output_dir=Path(str(data["output_dir"])),
        workers=data["workers"],
        threads_per_case=int(data["threads_per_case"]),
        retain_artifacts=bool(data["retain_artifacts"]),
        failure_policy=EFailurePolicy(str(data["failure_policy"])),
        storage=EWorkflowStorage(str(data["storage"])),
        max_in_flight=data["max_in_flight"],
        max_cases=data["max_cases"],
    )


def _slurm_script(run_dir: Path, config: SlurmConfig, case_count: int) -> str:
    """Return a self-contained Slurm job-array script for a prepared run."""
    array = f"0-{case_count - 1}"
    if config.array_concurrency is not None:
        array += f"%{config.array_concurrency}"
    directives = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=pyvale-workflow",
        f"#SBATCH --partition={config.partition}",
        f"#SBATCH --time={config.wall_time}",
        f"#SBATCH --cpus-per-task={config.cpus_per_task}",
        f"#SBATCH --array={array}",
        f"#SBATCH --output={run_dir}/slurm/%A_%a.out",
        f"#SBATCH --error={run_dir}/slurm/%A_%a.err",
    ]
    if config.account is not None:
        directives.append(f"#SBATCH --account={config.account}")
    if config.qos is not None:
        directives.append(f"#SBATCH --qos={config.qos}")
    if config.memory is not None:
        directives.append(f"#SBATCH --mem={config.memory}")
    commands = ["set -euo pipefail"]
    commands.extend(f"module load {module}" for module in config.modules)
    commands.extend(config.setup_commands)
    commands.extend((
        f"CASE_INDEX=$(sed -n \"$((SLURM_ARRAY_TASK_ID + 1))p\" "
        f"{run_dir}/slurm/case-indices.txt)",
        f"{config.python_executable} -m pyvale.workflow.worker "
        f"--run-dir {run_dir} --case-index \"${{CASE_INDEX}}\"",
    ))
    return "\n".join((*directives, "", *commands, ""))


def _json_default(value: object) -> object:
    """Convert stable parameter values into portable JSON representations."""
    return getattr(value, "value", str(value))
