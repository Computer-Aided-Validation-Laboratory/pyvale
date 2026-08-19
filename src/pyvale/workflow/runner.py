"""Serial and multiprocessing workflow execution."""

import json
import multiprocessing as mp
from multiprocessing.pool import AsyncResult, Pool
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from .case import WorkflowCase
from .config import EFailurePolicy, EWorkflowStorage, WorkflowConfig
from .errors import WorkflowCaseError
from .gather import WorkflowGatherer
from .pipeline import IWorkflow, WorkflowContext
from .result import CaseResult, ECaseStatus


def _json_default(value: object) -> object:
    """Convert stable scalar values for workflow manifests."""
    return getattr(value, "value", str(value))


def _run_case(
    workflow: IWorkflow,
    config: WorkflowConfig,
    case: WorkflowCase,
) -> CaseResult:
    """Run and persist one case in a worker-safe private directory."""
    case_dir = config.output_dir / "cases" / f"{case.index:06d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    persist_manifests = config.storage is not EWorkflowStorage.MEMORY
    if persist_manifests:
        parameters_path = case_dir / "parameters.json"
        with parameters_path.open("w", encoding="utf-8") as file:
            json.dump(dict(case.values), file, indent=2, default=_json_default)
    context = WorkflowContext(config, case_dir, np.random.default_rng(case.seed))
    context.add_artifact(case_dir)
    start_time = perf_counter()
    try:
        result = workflow.run_case(case, context)
    except Exception as exception:
        result = CaseResult(
            case=case,
            metrics={},
            artifacts=tuple(context.artifacts),
            status=ECaseStatus.FAILED,
            elapsed_seconds=perf_counter() - start_time,
            error=f"{type(exception).__name__}: {exception}",
        )
    if persist_manifests:
        summary = {
            "status": result.status.value,
            "metrics": dict(result.metrics),
            "error": result.error,
            "elapsed_seconds": result.elapsed_seconds,
        }
        summary_path = case_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, default=_json_default)
    if config.storage is EWorkflowStorage.MEMORY:
        shutil.rmtree(case_dir)
        return CaseResult(
            case=result.case,
            metrics=result.metrics,
            artifacts=(),
            status=result.status,
            elapsed_seconds=result.elapsed_seconds,
            error=result.error,
        )
    if (config.storage is EWorkflowStorage.HYBRID
            and not config.retain_artifacts):
        _remove_large_artifacts(case_dir)
    return result


def _remove_large_artifacts(case_dir: Path) -> None:
    """Remove case files except compact parameters and summary manifests."""
    retained = {"parameters.json", "summary.json"}
    for path in case_dir.rglob("*"):
        if path.is_file() and path.name not in retained:
            path.unlink()


class WorkflowRunner:
    """Execute validated workflow cases serially or with multiprocessing."""

    def __init__(self, config: WorkflowConfig) -> None:
        """Store run configuration."""
        self.config = config

    def run(
        self,
        workflow: IWorkflow,
        cases: list[WorkflowCase],
    ) -> tuple[CaseResult, ...]:
        """Validate, execute, persist, and return ordered case results."""
        if self.config.max_cases is not None and len(cases) > self.config.max_cases:
            raise ValueError("Workflow case count exceeds WorkflowConfig.max_cases.")
        if self.config.workers is not None and self.config.workers <= 0:
            raise ValueError("WorkflowConfig.workers must be positive or None.")
        if self.config.max_in_flight is not None and self.config.max_in_flight <= 0:
            raise ValueError("WorkflowConfig.max_in_flight must be positive or None.")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        for case in cases:
            workflow.verify_case(case)
        workers = 1 if self.config.workers is None else self.config.workers
        if workers == 1:
            results = [_run_case(workflow, self.config, case) for case in cases]
        else:
            context = mp.get_context("spawn")
            with context.Pool(processes=workers) as pool:
                results = _run_cases_bounded(
                    pool,
                    workflow,
                    self.config,
                    cases,
                    self.config.max_in_flight or workers,
                )
        results = sorted(results, key=lambda result: result.case.index)
        if self.config.failure_policy is EFailurePolicy.RAISE:
            failed = next(
                (item for item in results if item.status is ECaseStatus.FAILED),
                None,
            )
            if failed is not None:
                raise WorkflowCaseError(
                    "Workflow case "
                    f"{failed.case.index} failed for {dict(failed.case.values)}: "
                    f"{failed.error}",
                )
        if self.config.storage is not EWorkflowStorage.MEMORY:
            manifest = {
                "case_count": len(results),
                "failed_count": sum(
                    (item.status is ECaseStatus.FAILED for item in results),
                ),
            }
            manifest_path = self.config.output_dir / "manifest.json"
            with manifest_path.open("w", encoding="utf-8") as file:
                json.dump(manifest, file, indent=2)
            dataset = WorkflowGatherer.from_results(tuple(results))
            np.savez(
                self.config.output_dir / "summary.npz",
                statuses=dataset.statuses,
                **{f"parameter__{key}": value
                   for key, value in dataset.parameters.items()},
                **{f"metric__{key}": value
                   for key, value in dataset.metrics.items()},
            )
        return tuple(results)


def _run_cases_bounded(
    pool: Pool,
    workflow: IWorkflow,
    config: WorkflowConfig,
    cases: list[WorkflowCase],
    max_in_flight: int,
) -> list[CaseResult]:
    """Submit at most ``max_in_flight`` independent process jobs at once."""
    pending: list[AsyncResult] = []
    results: list[CaseResult] = []
    case_iterator = iter(cases)

    def submit_next() -> bool:
        """Submit one remaining case and report whether one was available."""
        try:
            case = next(case_iterator)
        except StopIteration:
            return False
        pending.append(pool.apply_async(_run_case, (workflow, config, case)))
        return True

    while len(pending) < max_in_flight and submit_next():
        pass
    while pending:
        results.append(pending.pop(0).get())
        submit_next()
    return results
