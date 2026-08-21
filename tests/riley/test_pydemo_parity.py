"""Parity checks between packaged Riley demos and Pyvale render examples."""

from __future__ import annotations

import runpy
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = _ROOT / "src" / "pyvale" / "examples" / "render"
_CASES = (
    ("demo_sphere200", "ex1b_riley_sphere200.py"),
    ("demo_psf", "ex1c_riley_psf.py"),
    ("demo_rabbits", "ex1d_riley_rabbits.py"),
    ("demo_dicuq", "ex1e_riley_dicuq.py"),
    ("demo_dic_from_exodus", "ex1f_riley_dic_from_exodus.py"),
    ("demo_stereocal", "ex1g_riley_stereocal.py"),
)


@pytest.fixture
def demo_directories(tmp_path: Path) -> Iterator[tuple[Path, Path]]:
    """Provide isolated native and Pyvale demo roots, then remove renders."""
    native_dir = tmp_path / "native"
    pyvale_dir = tmp_path / "pyvale"
    native_dir.mkdir()
    pyvale_dir.mkdir()
    try:
        yield native_dir, pyvale_dir
    finally:
        shutil.rmtree(native_dir, ignore_errors=True)
        shutil.rmtree(pyvale_dir, ignore_errors=True)


@pytest.mark.riley
@pytest.mark.example_slow
@pytest.mark.parametrize(("native_name", "example_name"), _CASES)
def test_packaged_demo_matches_pyvale_example(
    demo_directories: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    native_name: str,
    example_name: str,
) -> None:
    """Render each installed demo natively and through the public API."""
    native_dir, pyvale_dir = demo_directories
    _run_native(native_name, native_dir, monkeypatch)
    _run_example(example_name, pyvale_dir, monkeypatch)
    _assert_same_tree(native_dir / "out-riley-py", pyvale_dir / "out-riley-py")


def _run_native(
    demo_name: str,
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run a packaged demo from the installed Riley dependency."""
    import importlib

    monkeypatch.chdir(directory)
    if demo_name == "demo_stereocal":
        importlib.import_module("riley.pydemos.demo_dicuq").main()
    importlib.import_module(f"riley.pydemos.{demo_name}").main()


def _run_example(
    example_name: str,
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the matching documented Pyvale example in an isolated directory."""
    monkeypatch.chdir(directory)
    sys.path.insert(0, str(_EXAMPLES))
    try:
        if example_name == "ex1g_riley_stereocal.py":
            runpy.run_path(str(_EXAMPLES / "ex1e_riley_dicuq.py"))
        runpy.run_path(str(_EXAMPLES / example_name))
    finally:
        sys.path.remove(str(_EXAMPLES))


def _assert_same_tree(native: Path, pyvale: Path) -> None:
    """Require byte-identical render and calibration outputs."""
    native_paths = sorted(path.relative_to(native) for path in native.rglob("*"))
    pyvale_paths = sorted(path.relative_to(pyvale) for path in pyvale.rglob("*"))
    assert native_paths == pyvale_paths
    for relative_path in native_paths:
        native_path = native / relative_path
        if native_path.is_file():
            assert native_path.read_bytes() == (pyvale / relative_path).read_bytes()
