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
        "demo_sphere200",
        "ex1b_riley_sphere200.py",
        "out-riley-py/demo-sphere200",
        "pyvale-output/render-riley-sphere200",
        id="sphere200",
    ),
    pytest.param(
        "demo_psf",
        "ex1c_riley_psf.py",
        "out-riley-py/demo-psf",
        "pyvale-output/render-riley-psf",
        id="psf",
    ),
    pytest.param(
        "demo_rabbits",
        "ex1d_riley_rabbits.py",
        "out-riley-py/demo-rabbits",
        "pyvale-output/render-riley-rabbits",
        id="rabbits",
    ),
    pytest.param(
        "demo_dicuq",
        "ex1e_riley_dicuq.py",
        "out-riley-py/demo-dicuq",
        "pyvale-output/render-riley-dicuq",
        id="dicuq",
    ),
    pytest.param(
        "demo_dic_from_exodus",
        "ex1f_riley_dic_from_exodus.py",
        "out-riley-py/demo-dicuq-from-exodus",
        "pyvale-output/render-riley-exodus",
        id="dicuq-from-exodus",
    ),
    pytest.param(
        "demo_stereocal",
        "ex1g_riley_stereocal.py",
        "out-riley-py/demo-stereocal",
        "pyvale-output/render-riley-stereocal",
        id="stereocal",
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
@pytest.mark.parametrize(("demo", "example", "native_out", "pyvale_out"), _CASES)
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
    if demo == "demo_stereocal":
        # The stereo-calibration demo consumes the dicuq demo's stereo pair.
        importlib.import_module("riley.pydemos.demo_dicuq").main()
    importlib.import_module(f"riley.pydemos.{demo}").main()

    monkeypatch.chdir(pyvale_dir)
    if example == "ex1g_riley_stereocal.py":
        # The stereo-calibration example consumes the ex1e example's outputs.
        runpy.run_path(str(_EXAMPLES / "ex1e_riley_dicuq.py"))
    runpy.run_path(str(_EXAMPLES / example))

    _assert_same_tree(
        native_dir / native_out,
        pyvale_dir / pyvale_out,
    )


def _assert_same_tree(native: Path, pyvale: Path) -> None:
    """Require byte-identical render outputs in both output trees."""
    assert native.is_dir(), f"native render output missing: {native}"
    assert pyvale.is_dir(), f"pyvale render output missing: {pyvale}"

    native_paths = sorted(path.relative_to(native) for path in native.rglob("*"))
    pyvale_paths = sorted(path.relative_to(pyvale) for path in pyvale.rglob("*"))
    assert native_paths == pyvale_paths
    for relative_path in native_paths:
        native_path = native / relative_path
        if native_path.is_file():
            assert native_path.read_bytes() == (pyvale / relative_path).read_bytes(), (
                f"render output differs: {relative_path}"
            )
