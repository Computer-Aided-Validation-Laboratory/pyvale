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
        parameters={"n": 5},
        output_dir=tmp_path / "mesh",
    )
    result = mh.MooseRunner().run(
        input_tree / "mesh_model.i",
        parameters={"Mesh/file": str(mesh.output)},
        output_dir=tmp_path / "solve",
    )
    from netCDF4 import Dataset

    with Dataset(result.output) as data:
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
    (tree / "bad.py").write_text('raise ValueError("bad mesh parameters")\n')
    with pytest.raises(RuntimeError, match="run.log"):
        mh.GmshRunner().run(tree / "bad.py", output_dir=tmp_path / "mesh")
    assert "bad mesh parameters" in (tmp_path / "mesh/run.log").read_text()


def test_custom_gmsh_arguments(input_tree: Path, tmp_path: Path, monkeypatch) -> None:
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        (Path(kwargs["cwd"]) / "mesh.msh").write_bytes(b"mesh")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("pyvale.mooseherder.simrunner.subprocess.run", fake_run)
    mh.GmshRunner(executable=Path(sys.executable), order=2).run(
        input_tree / "specimen.geo",
        parameters={"n": 5},
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
