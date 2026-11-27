#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

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
            mh.SimStep("mesh", mh.GmshRunner(), input_tree / "specimen.geo"),
            mh.SimStep(
                "solve", mh.MooseRunner(), input_tree / "mesh_model.i", mesh_from="mesh"
            ),
        ],
        output_dir=tmp_path / "sweeps",
        parallel_runners=parallel,
    )
    case_input_params = [
        {"mesh": {"n": 3}, "solve": {"topDisp": 1e-4}},
        {"mesh": {"n": 4}, "solve": {"topDisp": 2e-4}},
    ]
    results = herd.run(case_input_params)
    assert len({r.output_dir for row in results for r in row}) == 4
    manifest = tmp_path / "sweeps/sweep_0001/sweep.json"
    loader = mh.SweepLoader(manifest)
    assert loader.case_input_params == case_input_params
    simulations = loader.load_sim_data("solve")
    for data, displacement in zip(simulations, (1e-4, 2e-4)):
        np.testing.assert_allclose(
            data.node_vars["disp_y"][:, -1],
            displacement * data.coords[:, 1],
            atol=1e-12,
        )
    mesh_parameter = json.loads(
        (results[0][1].output_dir / "input_params.json").read_text()
    )["Mesh/file"]
    assert mesh_parameter == str(results[0][0].output_file)
    herd.run(case_input_params[:1])
    assert (tmp_path / "sweeps/sweep_0002/sweep.json").is_file()


def test_failed_mesh_stops_chain(input_tree: Path, tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "bad.py").write_text('raise RuntimeError("mesh failed")\n')
    herd = mh.MooseHerd(
        [
            mh.SimStep("mesh", mh.GmshRunner(), tree / "bad.py"),
            mh.SimStep(
                "solve", mh.MooseRunner(), input_tree / "mesh_model.i", mesh_from="mesh"
            ),
        ],
        output_dir=tmp_path / "sweeps",
    )
    with pytest.raises(RuntimeError, match="run.log"):
        herd.run([{}])
    assert not (tmp_path / "sweeps/sweep_0001/run_0001/solve").exists()
    assert not (tmp_path / "sweeps/sweep_0001/sweep.json").exists()


def test_step_validation(input_tree: Path, tmp_path: Path) -> None:
    step = mh.SimStep("solve", mh.MooseRunner(), input_tree / "model.i")
    with pytest.raises(ValueError, match="duplicate"):
        mh.MooseHerd([step, step], output_dir=tmp_path)
    with pytest.raises(ValueError, match="earlier"):
        mh.MooseHerd(
            [
                mh.SimStep(
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
    case_input_params = mh.sweep_param_grid({"mesh": {"n": [3, 4]}, "solve": {"EMod": [1e9, 2e9]}})
    assert len(case_input_params) == 4
    assert case_input_params[0] == {"mesh": {"n": 3}, "solve": {"EMod": 1e9}}
    assert case_input_params[-1] == {"mesh": {"n": 4}, "solve": {"EMod": 2e9}}
    with pytest.raises(ValueError):
        mh.sweep_param_grid({"mesh": {"n": []}})


def test_reuse_real_solves_and_failed_replacement(
    input_tree: Path, tmp_path: Path
) -> None:
    manager = mh.DirectoryManager(tmp_path / "particles", mode=mh.EDirMode.REUSE)
    herd = mh.MooseHerd(
        [mh.SimStep("solve", mh.MooseRunner(), input_tree / "model.i")],
        dir_manager=manager,
        parallel_runners=2,
    )
    first = herd.run(
        [
            {"solve": {"topDisp": 1e-4}},
            {"solve": {"topDisp": 2e-4}},
        ],
        reuse_ids=["particle_1", "particle_7"],
    )
    first_particle = first[0][0].output_file.read_bytes()
    loader = mh.SweepLoader(manager.output_dir / "particle_7/evaluation.json")
    assert loader.case_input_params[0]["solve"]["topDisp"] == 2e-4
    second = herd.run(
        [
            {"solve": {"topDisp": 3e-4}},
        ],
        reuse_ids=["particle_7"],
    )
    assert first[1][0].output_file == second[0][0].output_file
    assert first[0][0].output_file.read_bytes() == first_particle
    loader = mh.SweepLoader(manager.output_dir / "particle_7/evaluation.json")
    data = loader.load_sim_data("solve")[0]
    np.testing.assert_allclose(
        data.node_vars["disp_y"][:, -1], 3e-4 * data.coords[:, 1], atol=1e-12
    )
    with pytest.raises(RuntimeError):
        herd.run([{"solve": {"nonexistent": 1}}], reuse_ids=["particle_7"])
    assert not second[0][0].output_file.exists()
    assert not (manager.output_dir / "particle_7/evaluation.json").exists()
    assert (manager.output_dir / "particle_7/solve/run.log").is_file()
    # A failure releases the reservation and does not prevent a fresh attempt.
    herd.run([{}], reuse_ids=["particle_7"])
    assert (manager.output_dir / "particle_7/evaluation.json").is_file()
    assert not list(manager.output_dir.glob("sweep_*"))


def test_reuse_does_not_delete_source_tree(input_tree: Path, tmp_path: Path) -> None:
    import shutil

    root = tmp_path / "particles"
    source = root / "particle_1"
    shutil.copytree(input_tree, source)
    manager = mh.DirectoryManager(root, mode=mh.EDirMode.REUSE)
    with pytest.raises(ValueError, match="Input trees"):
        mh.MooseHerd(
            [
                mh.SimStep("solve", mh.MooseRunner(), source / "model.i"),
            ],
            dir_manager=manager,
        )
    assert (source / "model.i").is_file()


def test_archive_can_replay_an_existing_input_snapshot(
    input_tree: Path, tmp_path: Path
) -> None:
    import shutil

    output_root = tmp_path / "archive"
    source = output_root / "saved-input_files"
    shutil.copytree(input_tree, source)
    original = (source / "model.i").read_bytes()
    herd = mh.MooseHerd(
        [
            mh.SimStep("solve", mh.MooseRunner(), source / "model.i"),
        ],
        output_dir=output_root,
    )
    herd.run([{}])
    assert (output_root / "sweep_0001/sweep.json").is_file()
    assert (source / "model.i").read_bytes() == original
