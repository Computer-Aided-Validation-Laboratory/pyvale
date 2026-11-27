#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

"""Portable completed sweep manifests."""

import shutil
from pathlib import Path

import pyvale.mooseherder as mh


def test_relocated_sweep(input_tree: Path, tmp_path: Path) -> None:
    herd = mh.MooseHerd(
        [
            mh.SimStep("solve", mh.MooseRunner(), input_tree / "model.i"),
        ],
        output_dir=tmp_path / "original",
    )
    herd.run([{}])
    shutil.copytree(tmp_path / "original/sweep_0001", tmp_path / "relocated")
    loader = mh.SweepLoader(tmp_path / "relocated/sweep.json")
    assert loader.outputs[0][0] == tmp_path / "relocated/run_0001/solve/results.e"
    assert len(loader.load_sim_data("solve")) == 1
