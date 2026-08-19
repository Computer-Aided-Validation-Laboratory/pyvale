"""Workflow result containers."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from .case import WorkflowCase


class ECaseStatus(Enum):
    """Terminal state of a workflow case."""

    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Compact outcome from one workflow case."""

    case: WorkflowCase
    metrics: Mapping[str, float]
    artifacts: tuple[Path, ...]
    status: ECaseStatus
    elapsed_seconds: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class WorkflowDataset:
    """Column-oriented gathered workflow data."""

    parameters: Mapping[str, np.ndarray]
    metrics: Mapping[str, np.ndarray]
    statuses: np.ndarray
    case_dirs: tuple[Path, ...]

