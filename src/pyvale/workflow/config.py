"""Workflow execution configuration."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class EFailurePolicy(Enum):
    """Action taken after one workflow case fails."""

    CONTINUE = "continue"
    RAISE = "raise"


class EWorkflowStorage(Enum):
    """Persistence strategy for workflow artifacts.

    ``DISK`` retains all case data, ``HYBRID`` retains compact manifests and
    applies :attr:`WorkflowConfig.retain_artifacts` to large files, and
    ``MEMORY`` removes all per-case data after each case completes.
    """

    DISK = "disk"
    MEMORY = "memory"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    """Execution and persistence controls for one workflow run.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory for persisted manifests and case work directories.
    workers : int or None, optional
        Number of independent worker processes. ``None`` selects serial work.
    threads_per_case : int, optional
        Per-case thread budget supplied to workflow functions when needed.
    retain_artifacts : bool, optional
        Retain large artifacts in :attr:`EWorkflowStorage.HYBRID` mode.
    failure_policy : EFailurePolicy, optional
        Whether failed cases are reported or raised after all cases finish.
    storage : EWorkflowStorage, optional
        Persistence strategy for workflow case data.
    max_in_flight, max_cases : int or None, optional
        Limits for future bounded dispatch and the accepted case count.
    """

    output_dir: Path
    workers: int | None = None
    threads_per_case: int = 1
    retain_artifacts: bool = True
    failure_policy: EFailurePolicy = EFailurePolicy.CONTINUE
    storage: EWorkflowStorage = EWorkflowStorage.HYBRID
    max_in_flight: int | None = None
    max_cases: int | None = None

    def __post_init__(self) -> None:
        """Normalise the output directory."""
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.workers is not None and self.workers <= 0:
            raise ValueError("workers must be positive or None.")
        if self.threads_per_case <= 0:
            raise ValueError("threads_per_case must be positive.")
        if self.max_in_flight is not None and self.max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive or None.")
        if self.max_cases is not None and self.max_cases <= 0:
            raise ValueError("max_cases must be positive or None.")
