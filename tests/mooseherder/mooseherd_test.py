"""Real sequential/parallel chains, result ordering and failure propagation."""

import json
from pathlib import Path

import numpy as np
import pytest

import pyvale.mooseherder as mh


@pytest.mark.parametrize("parallel", [1, 2])
def test_chain_sweep(input_tree: Path, tmp_path: Path, parallel: int) -> None:
    herd = mh.MooseHerd(
        [
            mh.SimulationStep("mesh", mh.GmshRunner(), input_tree / "specimen.geo"),
            mh.SimulationStep(
                "solve", mh.MooseRunner(), input_tree / "mesh_model.i", mesh_from="mesh"
            ),
        ],
        output_dir=tmp_path / "sweeps",
        num_parallel=parallel,
    )
    cases = [
        {"mesh": {"n": 3}, "solve": {"topDisp": 1e-4}},
        {"mesh": {"n": 4}, "solve": {"topDisp": 2e-4}},
    ]
    results = herd.run(cases)
    assert len({r.directory for row in results for r in row}) == 4
    manifest = tmp_path / "sweeps/sweep-0001/sweep.json"
    loader = mh.SweepLoader(manifest)
    assert loader.cases == cases
    simulations = loader.load_sim_data("solve")
    for data, displacement in zip(simulations, (1e-4, 2e-4)):
        np.testing.assert_allclose(
            data.node_vars["disp_y"][:, -1],
            displacement * data.coords[:, 1],
            atol=1e-12,
        )
    mesh_parameter = json.loads(
        (results[0][1].directory / "parameters.json").read_text()
    )["Mesh/file"]
    assert mesh_parameter == str(results[0][0].output)
    herd.run(cases[:1])
    assert (tmp_path / "sweeps/sweep-0002/sweep.json").is_file()


def test_failed_mesh_stops_chain(input_tree: Path, tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "bad.py").write_text('raise RuntimeError("mesh failed")\n')
    herd = mh.MooseHerd(
        [
            mh.SimulationStep("mesh", mh.GmshRunner(), tree / "bad.py"),
            mh.SimulationStep(
                "solve", mh.MooseRunner(), input_tree / "mesh_model.i", mesh_from="mesh"
            ),
        ],
        output_dir=tmp_path / "sweeps",
    )
    with pytest.raises(RuntimeError, match="run.log"):
        herd.run([{}])
    assert not (tmp_path / "sweeps/sweep-0001/run-0001/solve").exists()
    assert not (tmp_path / "sweeps/sweep-0001/sweep.json").exists()


def test_step_validation(input_tree: Path, tmp_path: Path) -> None:
    step = mh.SimulationStep("solve", mh.MooseRunner(), input_tree / "model.i")
    with pytest.raises(ValueError, match="duplicate"):
        mh.MooseHerd([step, step], output_dir=tmp_path)
    with pytest.raises(ValueError, match="earlier"):
        mh.MooseHerd(
            [
                mh.SimulationStep(
                    "solve",
                    mh.MooseRunner(),
                    input_tree / "model.i",
                    mesh_from="missing",
                ),
            ],
            output_dir=tmp_path,
        )
    herd = mh.MooseHerd([step], output_dir=tmp_path / "sweeps")
    with pytest.raises(ValueError, match="Unknown"):
        herd.run([{"bad": {}}])
    assert not (tmp_path / "sweeps").exists()


def test_parameter_grid() -> None:
    cases = mh.sweep_param_grid({"mesh": {"n": [3, 4]}, "solve": {"EMod": [1e9, 2e9]}})
    assert len(cases) == 4
    assert cases[0] == {"mesh": {"n": 3}, "solve": {"EMod": 1e9}}
    assert cases[-1] == {"mesh": {"n": 4}, "solve": {"EMod": 2e9}}
    with pytest.raises(ValueError):
        mh.sweep_param_grid({"mesh": {"n": []}})


def test_reuse_real_solves_and_failed_replacement(
    input_tree: Path, tmp_path: Path
) -> None:
    manager = mh.DirectoryManager(tmp_path / "particles", mode=mh.EDirectoryMode.REUSE)
    herd = mh.MooseHerd(
        [mh.SimulationStep("solve", mh.MooseRunner(), input_tree / "model.i")],
        directory_manager=manager,
        num_parallel=2,
    )
    first = herd.run(
        [
            {"solve": {"topDisp": 1e-4}},
            {"solve": {"topDisp": 2e-4}},
        ],
        evaluation_ids=["particle-1", "particle-7"],
    )
    first_particle = first[0][0].output.read_bytes()
    loader = mh.SweepLoader(manager.output_dir / "particle-7/evaluation.json")
    assert loader.cases[0]["solve"]["topDisp"] == 2e-4
    second = herd.run(
        [
            {"solve": {"topDisp": 3e-4}},
        ],
        evaluation_ids=["particle-7"],
    )
    assert first[1][0].output == second[0][0].output
    assert first[0][0].output.read_bytes() == first_particle
    loader = mh.SweepLoader(manager.output_dir / "particle-7/evaluation.json")
    data = loader.load_sim_data("solve")[0]
    np.testing.assert_allclose(
        data.node_vars["disp_y"][:, -1], 3e-4 * data.coords[:, 1], atol=1e-12
    )
    with pytest.raises(RuntimeError):
        herd.run([{"solve": {"nonexistent": 1}}], evaluation_ids=["particle-7"])
    assert not second[0][0].output.exists()
    assert not (manager.output_dir / "particle-7/evaluation.json").exists()
    assert (manager.output_dir / "particle-7/solve/run.log").is_file()
    # A failure releases the reservation and does not prevent a fresh attempt.
    herd.run([{}], evaluation_ids=["particle-7"])
    assert (manager.output_dir / "particle-7/evaluation.json").is_file()
    assert not list(manager.output_dir.glob("sweep-*"))


def test_reuse_does_not_delete_source_tree(input_tree: Path, tmp_path: Path) -> None:
    import shutil

    root = tmp_path / "particles"
    source = root / "particle-1"
    shutil.copytree(input_tree, source)
    manager = mh.DirectoryManager(root, mode=mh.EDirectoryMode.REUSE)
    with pytest.raises(ValueError, match="Input trees"):
        mh.MooseHerd(
            [
                mh.SimulationStep("solve", mh.MooseRunner(), source / "model.i"),
            ],
            directory_manager=manager,
        )
    assert (source / "model.i").is_file()


def test_archive_can_replay_an_existing_input_snapshot(
    input_tree: Path, tmp_path: Path
) -> None:
    import shutil

    output_root = tmp_path / "archive"
    source = output_root / "saved-inputs"
    shutil.copytree(input_tree, source)
    original = (source / "model.i").read_bytes()
    herd = mh.MooseHerd(
        [
            mh.SimulationStep("solve", mh.MooseRunner(), source / "model.i"),
        ],
        output_dir=output_root,
    )
    herd.run([{}])
    assert (output_root / "sweep-0001/sweep.json").is_file()
    assert (source / "model.i").read_bytes() == original
