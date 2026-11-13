# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Parse every packaged SMP/MPI entry and solve representative SMP case_input_params."""

import shutil
from pathlib import Path

import pytest
import rabbit

import pyvale.data as dataset
import pyvale.mooseherder as mh

CASE_ROOT = dataset.sim_case_directory_path()
CASE_NAMES = tuple(p.name for p in sorted(CASE_ROOT.glob("case*.i")))


@pytest.fixture(scope="module")
def staged_cases(tmp_path_factory) -> Path:
    """Mesh historical geometries once, outside the installed input tree."""
    directory = tmp_path_factory.mktemp("packaged-case_input_params")
    tree = directory / "input_files"
    shutil.copytree(CASE_ROOT, tree)
    for geometry in sorted(CASE_ROOT.glob("*.geo")):
        result = mh.GmshRunner().run(
            geometry,
            output_dir=directory / f"mesh-{geometry.stem}",
        )
        shutil.copy2(result.output_file, tree / f"{geometry.stem}.msh")
    return tree


@pytest.mark.parametrize("name", CASE_NAMES)
def test_packaged_deck(name: str, staged_cases: Path) -> None:
    """MPI solver syntax can be checked without launching multiple ranks."""
    rabbit.run_rabbit(
        staged_cases / name, cwd=staged_cases, extra_args=["--check-input"]
    )


@pytest.mark.parametrize("name", ["case00_HEX8.i", "case18.i"])
def test_packaged_smp_solve(name: str, staged_cases: Path, tmp_path: Path) -> None:
    """Cover both 3D and 2D coupled thermal/mechanical solver behaviour."""
    result = mh.MooseRunner().run(staged_cases / name, output_dir=tmp_path / "solve")
    assert result.output_file.stat().st_size > 0
