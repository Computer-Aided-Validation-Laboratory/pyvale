# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================
"""Availability checks for optional MOOSE and Gmsh executables."""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil


@dataclass(frozen=True, slots=True)
class BackendAvailability:
    """Availability state and actionable diagnostic for one external backend.

    Parameters
    ----------
    available : bool
        Whether the executable is available and executable.
    executable : pathlib.Path or None
        Resolved executable path when one was found.
    reason : str or None
        Explanation supplied when ``available`` is ``False``.
    """

    available: bool
    executable: Path | None
    reason: str | None = None


def moose_availability(config: dict[str, Path | str]) -> BackendAvailability:
    """Check that the MOOSE application configured for a runner is executable."""
    app_path = Path(config["app_path"])
    app_name = Path(str(config["app_name"]))
    executable = app_name if app_name.is_absolute() else app_path / app_name
    return _availability_for_path("MOOSE", executable)


def gmsh_availability(gmsh_path: Path | None = None) -> BackendAvailability:
    """Find an explicit Gmsh executable or resolve ``gmsh`` from ``PATH``."""
    if gmsh_path is not None:
        return _availability_for_path("Gmsh", gmsh_path)
    resolved = shutil.which("gmsh")
    if resolved is None:
        return BackendAvailability(
            False,
            None,
            "Gmsh was not found on PATH. Install Gmsh or configure its path.",
        )
    return _availability_for_path("Gmsh", Path(resolved))


def _availability_for_path(name: str, executable: Path) -> BackendAvailability:
    """Return availability for one explicit executable path."""
    if not executable.is_file():
        return BackendAvailability(
            False,
            None,
            f"{name} executable was not found: {executable}",
        )
    if not os.access(executable, os.X_OK):
        return BackendAvailability(
            False,
            executable,
            f"{name} executable is not executable: {executable}",
        )
    return BackendAvailability(True, executable)
