"""Availability diagnostics for packaged defaults and explicit executables."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BackendAvailability:
    """Whether the selected backend can run, with an actionable diagnostic."""

    available: bool
    executable: Path | None
    reason: str | None = None


def _availability_for_path(name: str, executable: Path) -> BackendAvailability:
    if not executable.is_file():
        return BackendAvailability(
            False, None, f"{name} executable not found: {executable}"
        )
    if not os.access(executable, os.X_OK):
        return BackendAvailability(
            False, executable, f"{name} is not executable: {executable}"
        )
    return BackendAvailability(True, executable)


def moose_availability(
    executable: Path | None = None,
    *,
    mpi: bool = False,
) -> BackendAvailability:
    """Resolve Rabbit by default; never fall back from an explicit app path."""
    if executable is not None:
        return _availability_for_path("MOOSE", executable)
    try:
        from rabbit import get_binary_path
        from rabbit.cli import is_mpi_binary

        path = get_binary_path()
        if mpi and not is_mpi_binary():
            return BackendAvailability(
                False,
                path,
                "MPI requires an MPI-capable Rabbit installation; "
                "the installed rabbit-fem build is serial/SMP.",
            )
        return _availability_for_path("Rabbit", path)
    except (ImportError, OSError) as err:
        return BackendAvailability(False, None, f"Rabbit is unavailable: {err}")


def gmsh_availability(executable: Path | None = None) -> BackendAvailability:
    """Check the Python Gmsh library by default, or a custom executable."""
    if executable is not None:
        return _availability_for_path("Gmsh", executable)
    try:
        import gmsh

        return BackendAvailability(True, Path(gmsh.__file__).resolve())
    except (ImportError, OSError) as err:
        return BackendAvailability(False, None, f"Python Gmsh is unavailable: {err}")
