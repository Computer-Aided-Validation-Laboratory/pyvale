"""CLI, staging, errors and analytic mechanics with the installed Rabbit build."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

import pyvale.mooseherder as mh
from pyvale.mooseherder.simrunner import parameter_args


def test_parameter_serialization() -> None:
    assert parameter_args(
        {
            "EMod": 1e9,
            "Mesh/nx": 4,
            "active": True,
            "names": ("disp_x", "disp_y"),
            "function": "x + y",
        }
    ) == [
        "EMod=1000000000.0",
        "Mesh/nx=4",
        "active=true",
        "names=disp_x disp_y",
        "function=x + y",
    ]


@pytest.mark.parametrize(
    "parameters",
    [
        {"-bad": 1},
        {"a b": 1},
        {"a//b": 1},
        {"a": float("nan")},
        {"a": float("inf")},
        {"a": []},
        {"a": "x\ny"},
        {"a": [[1]]},
    ],
)
def test_invalid_parameters(parameters: dict) -> None:
    with pytest.raises((ValueError, TypeError)):
        parameter_args(parameters)


def test_cli_nested_includes_and_analytic_solution(
    input_tree: Path, tmp_path: Path
) -> None:
    original = (input_tree / "parameters.i").read_bytes()
    result = mh.MooseRunner().run(
        input_tree / "model.i",
        output_dir=tmp_path / "run with spaces",
        parameters={
            "EMod": 2e9,
            "topDisp": 2e-4,
            "nElemX": 3,
            "GlobalParams/displacements": ("disp_x", "disp_y"),
        },
    )
    assert (result.directory / "inputs/materials/elastic.i").is_file()
    assert (input_tree / "parameters.i").read_bytes() == original
    with Dataset(result.output) as nc:
        assert len(nc.dimensions["num_elem"]) == 6
    reader = mh.ExodusLoader(result.output)
    try:
        data = reader.load_all_sim_data()
    finally:
        reader.close()
    # Unit-square plane-strain solution with free lateral contraction.
    eps = 2e-4
    np.testing.assert_allclose(
        data.node_vars["disp_y"][:, -1], eps * data.coords[:, 1], atol=1e-12
    )
    np.testing.assert_allclose(
        data.node_vars["disp_x"][:, -1],
        -0.3 / 0.7 * eps * data.coords[:, 0],
        atol=1e-12,
    )
    stress = data.elem_vars[("stress_yy", 1)][:, -1]
    np.testing.assert_allclose(stress, 2e9 / (1 - 0.3**2) * eps, rtol=1e-8)
    assert json.loads((result.directory / "command.json").read_text()) == list(
        result.command
    )
    with pytest.raises(FileExistsError):
        mh.MooseRunner().run(input_tree / "model.i", output_dir=result.directory)


@pytest.mark.parametrize("sequence", [list, tuple])
def test_multiple_entry_decks(sequence, input_tree: Path, tmp_path: Path) -> None:
    inputs = sequence((input_tree / "model.i", input_tree / "case_overlay.i"))
    result = mh.MooseRunner().run(
        inputs,
        input_root=input_tree,
        output_dir=tmp_path / "merge",
        parameters={"EMod": 3e9},
    )
    reader = mh.ExodusLoader(result.output)
    try:
        data = reader.load_all_sim_data()
    finally:
        reader.close()
    np.testing.assert_allclose(
        data.elem_vars[("stress_yy", 1)][:, -1], 3e9 / 0.91 * 1e-4, rtol=1e-8
    )


def test_failed_solve_keeps_log(input_tree: Path, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="run.log"):
        mh.MooseRunner().run(
            input_tree / "model.i",
            parameters={"nonexistent": 3},
            output_dir=tmp_path / "failed",
        )
    assert "unused parameter" in (tmp_path / "failed/run.log").read_text()
    assert (tmp_path / "failed/parameters.json").is_file()


def test_missing_include(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "bad.i").write_text("!include missing.i\n")
    with pytest.raises(RuntimeError, match="run.log"):
        mh.MooseRunner().run(tree / "bad.i", output_dir=tmp_path / "failed")


def test_custom_executable_does_not_modify_environment(
    input_tree: Path, tmp_path: Path
) -> None:
    from rabbit import get_binary_path, get_library_dir

    before = dict(os.environ)
    result = mh.MooseRunner(
        executable=get_binary_path(),
        env={"LD_LIBRARY_PATH": str(get_library_dir())},
    ).run(input_tree / "model.i", output_dir=tmp_path / "custom")
    assert result.output.is_file()
    assert dict(os.environ) == before
    assert str(get_binary_path()) == result.command[0]


@pytest.mark.parametrize(
    "options",
    [
        {"n_tasks": 0},
        {"n_threads": 0},
        {"n_tasks": 2},
        {"timeout": 0},
    ],
)
def test_invalid_run_options(options: dict) -> None:
    with pytest.raises(ValueError):
        mh.MooseRunner(**options)


def test_serial_rabbit_rejects_mpi(
    input_tree: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("rabbit.cli.is_mpi_binary", lambda: False)
    with pytest.raises(RuntimeError, match="serial/SMP"):
        mh.MooseRunner(mpi=True, n_tasks=2).run(
            input_tree / "model_mpi.i",
            output_dir=tmp_path / "mpi",
        )
    assert not (tmp_path / "mpi").exists()


def test_mpi_command_uses_explicit_launcher(
    input_tree: Path, tmp_path: Path, monkeypatch
) -> None:
    from rabbit import get_binary_path

    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        (Path(kwargs["cwd"]) / "results.e").write_bytes(b"test output")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "pyvale.mooseherder.mooserunner.shutil.which", lambda name: sys.executable
    )
    monkeypatch.setattr("pyvale.mooseherder.simrunner.subprocess.run", fake_run)
    mh.MooseRunner(
        executable=get_binary_path(),
        mpi=True,
        n_tasks=2,
        launcher=("my-launcher", "-n"),
    ).run(input_tree / "model_mpi.i", output_dir=tmp_path / "mpi")
    assert commands[0][:3] == ["my-launcher", "-n", "2"]


def test_input_tree_boundaries(input_tree: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="input_root"):
        mh.MooseRunner().run(
            (input_tree / "model.i", input_tree / "case_overlay.i"),
            output_dir=tmp_path / "bad",
        )
    with pytest.raises(ValueError, match="outside"):
        mh.MooseRunner().run(
            input_tree / "model.i", output_dir=input_tree / "bad-output"
        )


def test_rabbit_named_output_parameter(tmp_path: Path) -> None:
    from rabbit.sims import EElemType, cube_thermomech_input_path

    result = mh.MooseRunner(output_parameter="simName").run(
        cube_thermomech_input_path(EElemType.HEX8),
        parameters={"endTime": 1},
        output_dir=tmp_path / "rabbit-cube",
    )
    assert result.output.is_file()


def test_next_run_path_preserves_results(tmp_path: Path) -> None:
    output = tmp_path / "run"
    assert mh.next_run_path(output) == output
    output.mkdir()
    (output / "keep.txt").write_text("previous result")
    (tmp_path / "run-2").mkdir()
    assert mh.next_run_path(output) == tmp_path / "run-3"
    assert (output / "keep.txt").read_text() == "previous result"
