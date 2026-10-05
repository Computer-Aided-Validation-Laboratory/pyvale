"""Tests for optional external backend availability diagnostics."""

from pathlib import Path

import pyvale.mooseherder as mooseherder


def test_missing_moose_reports_configured_executable(tmp_path: Path) -> None:
    """MOOSE availability reports a missing application without executing it."""
    availability = mooseherder.moose_availability({
        "main_path": tmp_path,
        "app_path": tmp_path,
        "app_name": "missing-moose",
    })

    assert not availability.available
    assert availability.executable is None
    assert "missing-moose" in str(availability.reason)


def test_missing_gmsh_reports_path_lookup(monkeypatch) -> None:
    """Gmsh availability gives an install/configuration diagnostic."""
    monkeypatch.setattr("pyvale.mooseherder.availability.shutil.which", lambda _: None)

    availability = mooseherder.gmsh_availability()

    assert not availability.available
    assert availability.executable is None
    assert "not found on PATH" in str(availability.reason)
