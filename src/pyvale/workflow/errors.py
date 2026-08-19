"""Workflow-specific exceptions."""


class WorkflowError(RuntimeError):
    """Base exception raised for a failed workflow execution."""


class WorkflowCaseError(WorkflowError):
    """Raised when a configured workflow case cannot be completed."""
