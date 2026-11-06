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
