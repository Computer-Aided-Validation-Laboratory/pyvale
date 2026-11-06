"""Packaged defaults and explicit installation diagnostics."""

from pathlib import Path

import pyvale.mooseherder as mh


def test_packaged_defaults_available() -> None:
    assert mh.moose_availability().available
    assert mh.gmsh_availability().available


def test_explicit_missing_executables_do_not_fall_back(tmp_path: Path) -> None:
    for availability in (mh.moose_availability, mh.gmsh_availability):
        state = availability(tmp_path / "missing")
        assert not state.available
        assert "missing" in state.reason
