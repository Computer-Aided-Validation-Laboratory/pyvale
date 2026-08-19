# %%
"""Slurm workflow dry run
======================

Prepare a Slurm job-array workflow without contacting a scheduler. The same
persisted manifests can be checked locally with ``LocalArrayExecutor`` before
changing ``submit`` to ``True`` on a cluster.
"""

from pathlib import Path

import pyvale.workflow as workflow


def _verify_input(
    case: workflow.WorkflowCase,
    context: workflow.WorkflowContext,
    input_data: None,
) -> None:
    """Verify the source stage's empty payload."""
    if input_data is not None:
        raise ValueError("The source step expects no input.")


def _calculate_metric(
    case: workflow.WorkflowCase,
    context: workflow.WorkflowContext,
    input_data: None,
) -> dict[str, float]:
    """Represent an inexpensive case-local calculation for this dry run."""
    return {"value": float(case.values["value"]) ** 2}


def build_workflow() -> workflow.IWorkflow:
    """Return the importable factory used independently by array workers."""
    return workflow.PipelineWorkflow((
        workflow.FunctionStep(_verify_input, _calculate_metric),
    ))


def run_example(output_dir: Path = Path("workflow-slurm-dry-run-output")) -> None:
    """Generate a script and manifests without invoking ``sbatch``."""
    design = workflow.ExplicitCases(({"value": 1}, {"value": 2}, {"value": 3}))
    config = workflow.WorkflowConfig(output_dir, threads_per_case=2)
    slurm = workflow.SlurmConfig(
        partition="compute",
        cpus_per_task=2,
        array_concurrency=2,
        modules=("python",),
        submit=False,
    )
    workflow.SlurmExecutor(slurm).prepare(
        "pyvale.examples.workflow.ex4_slurm_dry_run:build_workflow",
        design.generate(2026),
        config,
    )


if __name__ == "__main__":
    run_example()
