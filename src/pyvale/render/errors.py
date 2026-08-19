# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Rendering validation errors."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One input problem found before rendering begins."""

    path: str
    code: str
    message: str


class RenderInputError(ValueError):
    """Raised when a render request has one or more invalid inputs."""

    def __init__(self, issues: tuple[ValidationIssue, ...]) -> None:
        self.issues = issues
        text = "\n".join(
            f"{issue.path} [{issue.code}]: {issue.message}"
            for issue in issues
        )
        super().__init__(text)


__all__ = ["RenderInputError", "ValidationIssue"]
