# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ==============================================================================

"""Byte-level parity checks between pyvale examples and native Riley demos.

Each case renders one packaged Riley demo natively and then runs the matching
documented pyvale example, which rebuilds the same scene through the public
render API. The rendered output trees must contain identical files with
identical contents.
"""

from __future__ import annotations

import importlib
import runpy
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = _ROOT / "src" / "pyvale" / "examples" / "render3d"

_CASES = (
    pytest.param(
        "demo0_quickstart",
        "ex1a_riley_quickstart.py",
        "out_riley_py/demo0_quickstart",
        "pyvale-output/render3d_ex1a_riley_quickstart",
        id="quickstart",
    ),
    pytest.param(
        "demo1_sphere200",
        "ex1b_riley_sphere200.py",
        "out_riley_py/demo1_sphere200",
        "pyvale-output/render3d_ex1b_riley_sphere200",
        id="sphere200",
    ),
    pytest.param(
        "demo3_rabbits",
        "ex1c_riley_rabbits.py",
        "out_riley_py/demo3_rabbits",
        "pyvale-output/render3d_ex1c_riley_rabbits",
        id="rabbits",
    ),
    pytest.param(
        "demo6_dicuq",
        "ex1d_riley_dicuq.py",
        "out_riley_py/demo6_dicuq",
        "pyvale-output/render3d_ex1d_riley_dicuq",
        id="dicuq",
    ),
    pytest.param(
        "demo7_dic_from_exodus",
        "ex1e_riley_dic_from_exodus.py",
        "out_riley_py/demo7_dic_from_exodus",
        "pyvale-output/render3d_ex1e_riley_dic_from_exodus",
        id="dicuq-from-exodus",
    ),
    pytest.param(
        "demo8_stereocal",
        "ex1f_riley_stereocal.py",
        "out_riley_py/demo8_stereocal",
        "pyvale-output/render3d_ex1f_riley_stereocal",
        id="stereocal",
    ),
    pytest.param(
        "demo2_psf",
        "ex1g_riley_psf.py",
        "out_riley_py/demo2_psf",
        "pyvale-output/render3d_ex1g_riley_psf",
        id="psf",
    ),
    pytest.param(
        "demo9_feature_zoo",
        "ex1h_riley_feature_zoo.py",
        "out_riley_py/demo9_feature_zoo",
        "pyvale-output/render3d_ex1h_riley_feature_zoo",
        id="feature-zoo",
    ),
)


@pytest.fixture
def demo_directories(tmp_path: Path) -> Iterator[tuple[Path, Path]]:
    """Provide isolated native and pyvale roots, then remove renders."""
    native_dir = tmp_path / "native"
    pyvale_dir = tmp_path / "pyvale"
    native_dir.mkdir()
    pyvale_dir.mkdir()
    yield native_dir, pyvale_dir
    shutil.rmtree(native_dir, ignore_errors=True)
    shutil.rmtree(pyvale_dir, ignore_errors=True)


@pytest.mark.riley
@pytest.mark.parametrize(
    ("demo", "example", "native_out", "pyvale_out"),
    _CASES,
)
def test_packaged_demo_matches_pyvale_example(
    demo_directories: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    demo: str,
    example: str,
    native_out: str,
    pyvale_out: str,
) -> None:
    """Render one demo natively and through the documented pyvale example."""
    native_dir, pyvale_dir = demo_directories

    monkeypatch.chdir(native_dir)
    importlib.import_module(f"riley.pydemos.{demo}").main()

    monkeypatch.chdir(pyvale_dir)
    runpy.run_path(str(_EXAMPLES / example))

    _assert_same_tree(
        native_dir / native_out,
        pyvale_dir / pyvale_out,
    )


def _assert_same_tree(native: Path, pyvale: Path) -> None:
    """Require byte-identical render outputs in both output trees."""
    assert native.is_dir(), f"native render output missing: {native}"
    assert pyvale.is_dir(), f"pyvale render output missing: {pyvale}"

    native_paths = sorted(
        path.relative_to(native) for path in native.rglob("*")
    )
    pyvale_paths = sorted(
        path.relative_to(pyvale) for path in pyvale.rglob("*")
    )
    assert native_paths == pyvale_paths
    for relative_path in native_paths:
        native_path = native / relative_path
        if native_path.is_file():
            assert (
                native_path.read_bytes()
                == (pyvale / relative_path).read_bytes()
            ), f"render output differs: {relative_path}"
