# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Backend checks fail at construction and are repeated before execution."""

from pathlib import Path

import pytest

import pyvale.mooseherder as mh


def test_packaged_backends_resolve() -> None:
    for runner in (mh.MooseRunner(), mh.GmshRunner()):
        backend = runner.get_backend()
        assert isinstance(backend, mh.RunnerBackend)
        assert backend.backend_file.is_absolute()
        assert backend.backend_file.is_file()


@pytest.mark.parametrize(
    "runner_type, exe_param",
    [(mh.MooseRunner, "moose_exe_file"), (mh.GmshRunner, "gmsh_exe_file")],
)
def test_missing_backend_fails_at_construction(
    tmp_path: Path, runner_type: type[mh.SimRunner], exe_param: str
) -> None:
    with pytest.raises(FileNotFoundError, match="missing"):
        runner_type(**{exe_param: tmp_path / "missing"})


@pytest.mark.parametrize(
    "runner_type, exe_param, entry_file",
    [
        (mh.MooseRunner, "moose_exe_file", "model.i"),
        (mh.GmshRunner, "gmsh_exe_file", "specimen.geo"),
    ],
)
def test_backend_is_rechecked_before_staging(
    tmp_path: Path,
    input_tree: Path,
    runner_type: type[mh.SimRunner],
    exe_param: str,
    entry_file: str,
) -> None:
    exe_file = tmp_path / "backend"
    exe_file.write_text("temporary backend")
    exe_file.chmod(0o755)
    runner = runner_type(**{exe_param: exe_file})
    exe_file.unlink()
    with pytest.raises(FileNotFoundError):
        runner.run(input_tree / entry_file, output_dir=tmp_path / "run")
    assert not (tmp_path / "run").exists()
