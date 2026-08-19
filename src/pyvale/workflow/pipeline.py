"""Generic typed linear workflow composition."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Generic, TypeVar

import numpy as np

from .case import WorkflowCase
from .config import WorkflowConfig
from .result import CaseResult, ECaseStatus

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


@dataclass(slots=True)
class WorkflowContext:
    """Mutable case-local resources supplied to workflow functions."""

    config: WorkflowConfig
    case_dir: Path
    rng: np.random.Generator
    artifacts: list[Path] = field(default_factory=list)

    def add_artifact(self, path: Path) -> None:
        """Record one retained case artifact."""
        self.artifacts.append(path)


class IWorkflowStep(ABC, Generic[TInput, TOutput]):
    """One validated transformation in a linear workflow pipeline."""

    @abstractmethod
    def verify_input(
        self,
        case: WorkflowCase,
        context: WorkflowContext,
        input_data: TInput,
    ) -> None:
        """Validate input before this step performs expensive work."""

    @abstractmethod
    def run(
        self,
        case: WorkflowCase,
        context: WorkflowContext,
        input_data: TInput,
    ) -> TOutput:
        """Run one validated workflow step."""


@dataclass(slots=True)
class FunctionStep(IWorkflowStep[TInput, TOutput]):
    """Adapt typed Python functions to one generic workflow step."""

    verify_function: Callable[[WorkflowCase, WorkflowContext, TInput], None]
    run_function: Callable[[WorkflowCase, WorkflowContext, TInput], TOutput]

    def verify_input(
        self,
        case: WorkflowCase,
        context: WorkflowContext,
        input_data: TInput,
    ) -> None:
        """Invoke the supplied cheap validation function."""
        self.verify_function(case, context, input_data)

    def run(
        self,
        case: WorkflowCase,
        context: WorkflowContext,
        input_data: TInput,
    ) -> TOutput:
        """Invoke the supplied run function."""
        return self.run_function(case, context, input_data)


class IWorkflow(ABC):
    """A reproducible study that evaluates independent workflow cases."""

    @abstractmethod
    def verify_case(self, case: WorkflowCase) -> None:
        """Validate case values before worker submission."""

    @abstractmethod
    def run_case(
        self,
        case: WorkflowCase,
        context: WorkflowContext,
    ) -> CaseResult:
        """Evaluate one case."""


class PipelineWorkflow(IWorkflow):
    """Compose generic function-backed steps into a linear workflow."""

    def __init__(
        self,
        steps: Sequence[IWorkflowStep[Any, Any]],
        verify_case_function: Callable[[WorkflowCase], None] | None = None,
    ) -> None:
        """Store pipeline steps and optional case validation."""
        self.steps = tuple(steps)
        self.verify_case_function = verify_case_function

    def verify_case(self, case: WorkflowCase) -> None:
        """Validate a case with the optional user-supplied function."""
        if self.verify_case_function is not None:
            self.verify_case_function(case)

    def run_case(
        self,
        case: WorkflowCase,
        context: WorkflowContext,
    ) -> CaseResult:
        """Run all steps and convert final metrics into a case result."""
        start_time = perf_counter()
        output: Any = None
        for step in self.steps:
            step.verify_input(case, context, output)
            output = step.run(case, context, output)
        if isinstance(output, CaseResult):
            return output
        if not isinstance(output, dict):
            raise TypeError(
                "The final workflow step must return metrics or CaseResult.",
            )
        return CaseResult(
            case=case,
            metrics={key: float(value) for key, value in output.items()},
            artifacts=tuple(context.artifacts),
            status=ECaseStatus.COMPLETED,
            elapsed_seconds=perf_counter() - start_time,
        )
