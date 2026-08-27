# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Rendering validation errors."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One input problem found before rendering begins.

    Parameters
    ----------
    path : str
        Dotted path to the invalid input.
    code : str
        Stable, machine-readable issue category.
    message : str
        Human-readable explanation of the problem.
    """

    path: str
    code: str
    message: str


class RenderInputError(ValueError):
    """Raised when a render request has one or more invalid inputs.

    Parameters
    ----------
    issues : tuple[ValidationIssue, ...]
        All validation issues detected for one render request.

    Attributes
    ----------
    issues : tuple[ValidationIssue, ...]
        All validation issues supplied at construction.
    """

    def __init__(self, issues: tuple[ValidationIssue, ...]) -> None:
        """Create an error whose text combines all validation issues."""
        self.issues = issues
        text = "\n".join(
            f"{issue.path} [{issue.code}]: {issue.message}" for issue in issues
        )
        super().__init__(text)


__all__ = ["RenderInputError", "ValidationIssue"]
