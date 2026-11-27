# ===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ===============================================================================

"""Real Gmsh Python geometry/script meshing and isolated failures."""

import subprocess
import sys
from pathlib import Path

import pytest

import pyvale.mooseherder as mh


@pytest.mark.parametrize("source", ["specimen.geo", "mesh_specimen.py"])
def test_mesh_source_and_solve(source: str, input_tree: Path, tmp_path: Path) -> None:
    mesh = mh.GmshRunner().run(
        input_tree / source,
        input_params={"n": 5},
        output_dir=tmp_path / "mesh",
    )
    result = mh.MooseRunner().run(
        input_tree / "mesh_model.i",
        input_params={"Mesh/file": str(mesh.output_file)},
        output_dir=tmp_path / "solve",
    )
    from netCDF4 import Dataset

    with Dataset(result.output_file) as data:
        assert len(data.dimensions["num_elem"]) == 16
        assert len(data.dimensions["num_nodes"]) == 25


def test_exit_in_geo_cannot_terminate_parent(input_tree: Path, tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "exit.geo").write_text("Exit;\n")
    with pytest.raises(RuntimeError, match="did not produce"):
        mh.GmshRunner().run(tree / "exit.geo", output_dir=tmp_path / "mesh")


def test_python_script_failure(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "bad.py").write_text('raise ValueError("bad mesh input_params")\n')
    with pytest.raises(RuntimeError, match="run.log"):
        mh.GmshRunner().run(tree / "bad.py", output_dir=tmp_path / "mesh")
    assert "bad mesh input_params" in (tmp_path / "mesh/run.log").read_text()


def test_custom_gmsh_arguments(input_tree: Path, tmp_path: Path, monkeypatch) -> None:
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        (Path(kwargs["cwd"]) / "mesh.msh").write_bytes(b"mesh")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("pyvale.mooseherder.simrunner.subprocess.run", fake_run)
    mh.GmshRunner(gmsh_exe_file=Path(sys.executable), mesh_order=2).run(
        input_tree / "specimen.geo",
        input_params={"n": 5},
        output_dir=tmp_path / "mesh",
    )
    assert commands[0][0] == str(Path(sys.executable).resolve())
    assert commands[0][-3:] == ["-setnumber", "n", "5"]
    assert "-order" in commands[0]


def test_bad_geo(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "bad.geo").write_text("this is not geometry;\n")
    with pytest.raises(RuntimeError, match="run.log"):
        mh.GmshRunner().run(tree / "bad.geo", output_dir=tmp_path / "mesh")


def test_script_without_output(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "empty.py").write_text('print("no mesh")\n')
    with pytest.raises(RuntimeError, match="did not produce"):
        mh.GmshRunner().run(tree / "empty.py", output_dir=tmp_path / "mesh")


def test_script_timeout_keeps_log(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "slow.py").write_text("import time\ntime.sleep(30)\n")
    with pytest.raises(RuntimeError, match="run.log"):
        mh.GmshRunner(timeout=0.1).run(tree / "slow.py", output_dir=tmp_path / "mesh")
    assert (tmp_path / "mesh/run.log").is_file()


@pytest.mark.parametrize("source", ["specimen.geo", "mesh_specimen.py"])
@pytest.mark.parametrize(
    "output_file_name,msh_version,expected_header",
    [
        ("custom.msh", 4.1, "$MeshFormat\n4.1"),
        ("custom.inp", 2.2, "*Heading"),
    ],
)
def test_export_format_and_version(
    source: str,
    output_file_name: str,
    msh_version: float,
    expected_header: str,
    input_tree: Path,
    tmp_path: Path,
) -> None:
    mesh = mh.GmshRunner(
        output_file_name=output_file_name, msh_version=msh_version
    ).run(input_tree / source, output_dir=tmp_path / "export")
    assert mesh.output_file.name == output_file_name
    assert expected_header.lower() in mesh.output_file.read_text().lower()


@pytest.mark.parametrize(
    "options",
    [
        {"output_file_name": "../escape.msh"},
        {"output_file_name": "subdir\\escape.msh"},
        {"output_file_name": "mesh"},
        {"output_file_name": "run.log"},
        {"output_file_name": "mesh_config.json"},
        {"msh_version": 0},
        {"msh_version": float("nan")},
        {"msh_version": float("inf")},
    ],
)
def test_invalid_output_settings(options: dict) -> None:
    with pytest.raises(ValueError):
        mh.GmshRunner(**options)


def test_custom_executable_format_settings(
    input_tree: Path, tmp_path: Path, monkeypatch
):
    def fake_run(command, **kwargs):
        assert command[command.index("-format") + 1] == "auto"
        version_index = command.index("Mesh.MshFileVersion")
        assert command[version_index + 1] == "4.1"
        assert Path(command[command.index("-o") + 1]).name == "custom.msh"
        (Path(kwargs["cwd"]) / "custom.msh").write_bytes(b"mesh")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("pyvale.mooseherder.simrunner.subprocess.run", fake_run)
    mh.GmshRunner(
        gmsh_exe_file=Path(sys.executable),
        output_file_name="custom.msh",
        msh_version=4.1,
    ).run(input_tree / "specimen.geo", output_dir=tmp_path / "export")
