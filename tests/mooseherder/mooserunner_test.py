# ===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ===============================================================================

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
from pyvale.mooseherder.simrunner import conv_args_dict_to_list


def test_parameter_serialization() -> None:
    assert conv_args_dict_to_list(
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
    "input_params",
    [
        {"-bad": 1},
        {"a b": 1},
        {"a=b": 1},
        {"a": float("nan")},
        {"a": float("inf")},
        {"a": []},
        {"a": "x\ny"},
        {"a": [[1]]},
    ],
)
def test_invalid_parameters(input_params: dict) -> None:
    with pytest.raises((ValueError, TypeError)):
        conv_args_dict_to_list(input_params)


def test_cli_nested_includes_and_analytic_solution(
    input_tree: Path, tmp_path: Path
) -> None:
    original = (input_tree / "parameters.i").read_bytes()
    result = mh.MooseRunner().run(
        input_tree / "model.i",
        output_dir=tmp_path / "run with spaces",
        input_params={
            "EMod": 2e9,
            "topDisp": 2e-4,
            "nElemX": 3,
            "GlobalParams/displacements": ("disp_x", "disp_y"),
        },
    )
    assert (result.output_dir / "input_files/materials/elastic.i").is_file()
    assert (input_tree / "parameters.i").read_bytes() == original
    with Dataset(result.output_file) as nc:
        assert len(nc.dimensions["num_elem"]) == 6
    with mh.ExodusLoader(result.output_file) as reader:
        data = reader.load_all_sim_data()
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
    assert json.loads((result.output_dir / "command.json").read_text()) == list(
        result.command
    )
    with pytest.raises(FileExistsError):
        mh.MooseRunner().run(input_tree / "model.i", output_dir=result.output_dir)


@pytest.mark.parametrize("sequence", [list, tuple])
def test_multiple_entry_decks(sequence, input_tree: Path, tmp_path: Path) -> None:
    input_files = sequence((input_tree / "model.i", input_tree / "case_overlay.i"))
    result = mh.MooseRunner().run(
        input_files,
        input_root=input_tree,
        output_dir=tmp_path / "merge",
        input_params={"EMod": 3e9},
    )
    with mh.ExodusLoader(result.output_file) as reader:
        data = reader.load_all_sim_data()
    np.testing.assert_allclose(
        data.elem_vars[("stress_yy", 1)][:, -1], 3e9 / 0.91 * 1e-4, rtol=1e-8
    )


def test_failed_solve_keeps_log(input_tree: Path, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="run.log"):
        mh.MooseRunner().run(
            input_tree / "model.i",
            input_params={"nonexistent": 3},
            output_dir=tmp_path / "failed",
        )
    assert "unused parameter" in (tmp_path / "failed/run.log").read_text()
    assert (tmp_path / "failed/input_params.json").is_file()


def test_missing_include(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "bad.i").write_text("!include missing.i\n")
    with pytest.raises(RuntimeError, match="run.log"):
        mh.MooseRunner().run(tree / "bad.i", output_dir=tmp_path / "failed")


def test_custom_executable_does_not_modify_os_environment_variables(
    input_tree: Path, tmp_path: Path
) -> None:
    from rabbit import get_binary_path, get_library_dir

    original_env_vars = dict(os.environ)
    result = mh.MooseRunner(
        moose_exe_file=get_binary_path(),
        env_vars={"LD_LIBRARY_PATH": str(get_library_dir())},
    ).run(input_tree / "model.i", output_dir=tmp_path / "custom")
    assert result.output_file.is_file()
    assert dict(os.environ) == original_env_vars
    assert str(get_binary_path()) == result.command[0]


@pytest.mark.parametrize(
    "options",
    [
        {"n_tasks": 0},
        {"n_threads": 0},
        {"n_tasks": True},
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
        mh.MooseRunner(n_tasks=2).run(
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
        moose_exe_file=get_binary_path(),
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
        input_params={"endTime": 1},
        output_dir=tmp_path / "rabbit-cube",
    )
    assert result.output_file.is_file()


def test_next_run_path_preserves_results(tmp_path: Path) -> None:
    output = tmp_path / "run"
    assert mh.get_next_run_path(output) == output
    output.mkdir()
    (output / "keep.txt").write_text("previous result")
    assert mh.get_next_run_path(output) == tmp_path / "run_1"
    (tmp_path / "run_1").mkdir()
    (tmp_path / "run_2").mkdir()
    assert mh.get_next_run_path(output) == tmp_path / "run_3"
    assert (output / "keep.txt").read_text() == "previous result"


@pytest.mark.parametrize("n_tasks", [None, 1])
def test_serial_task_count_and_timeout_defaults(
    input_tree: Path, tmp_path: Path, monkeypatch, n_tasks: int | None
) -> None:
    """Both serial task settings bypass MPI and retain the one-hour timeout."""

    def fake_run(command, **kwargs):
        assert command[0] == sys.executable
        assert kwargs["timeout"] == 3600.0
        (Path(kwargs["cwd"]) / "results.e").write_bytes(b"test output")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("pyvale.mooseherder.simrunner.subprocess.run", fake_run)
    runner = mh.MooseRunner(n_tasks=n_tasks)
    result = runner.run(input_tree / "model.i", output_dir=tmp_path / "serial")
    assert result.output_file == result.output_dir / "results.e"
    assert result.log_file == result.output_dir / "run.log"
    assert mh.GmshRunner().timeout == 3600.0


def test_moose_input_files_required(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="input_files"):
        mh.MooseRunner().run(output_dir=tmp_path / "missing")


def test_parameter_name_grammar_is_owned_by_backend() -> None:
    assert conv_args_dict_to_list({"a//b": 2}) == ["a//b=2"]


@pytest.mark.parametrize("name", ["", "-flag", "x=y", "a\tb", "a\0b", 1])
def test_output_parameter_validation(name) -> None:
    with pytest.raises((ValueError, TypeError)):
        mh.MooseRunner(output_parameter=name)


def test_env_vars_are_copied_and_merged_for_child(
    input_tree: Path, tmp_path: Path, monkeypatch
) -> None:
    """Solver OS environment variable overrides leave Python process values intact."""
    monkeypatch.setenv("PYVALE_TEST_INHERITED", "parent")
    monkeypatch.setenv("PYVALE_TEST_OVERRIDE", "parent")
    env_vars = {"PYVALE_TEST_OVERRIDE": "child"}
    runner = mh.MooseRunner(env_vars=env_vars)
    env_vars["PYVALE_TEST_OVERRIDE"] = "changed after construction"

    def fake_run(command, **kwargs):
        assert kwargs["env"]["PYVALE_TEST_INHERITED"] == "parent"
        assert kwargs["env"]["PYVALE_TEST_OVERRIDE"] == "child"
        (Path(kwargs["cwd"]) / "results.e").write_bytes(b"test output")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("pyvale.mooseherder.simrunner.subprocess.run", fake_run)
    runner.run(input_tree / "model.i", output_dir=tmp_path / "env_vars")
    assert os.environ["PYVALE_TEST_OVERRIDE"] == "parent"
    assert env_vars == {"PYVALE_TEST_OVERRIDE": "changed after construction"}


def test_configurable_output_file_base(input_tree: Path, tmp_path: Path) -> None:
    input_params: dict[str, mh.InputValues] = {"EMod": 2e9}
    result = mh.MooseRunner(output_file_base="custom_results").run(
        input_tree / "model.i",
        input_params=input_params,
        output_dir=tmp_path / "custom",
    )
    assert result.output_file.name == "custom_results.e"
    assert result.output_file.is_file()
    assert "Outputs/file_base=custom_results" in result.command
    assert input_params == {"EMod": 2e9}


@pytest.mark.parametrize(
    "output_file_base", ["", "../escape", "subdir/name", "subdir\\name", ".", ".."]
)
def test_output_file_base_stays_in_run_directory(output_file_base: str) -> None:
    with pytest.raises(ValueError):
        mh.MooseRunner(output_file_base=output_file_base)
