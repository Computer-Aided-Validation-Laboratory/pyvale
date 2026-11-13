# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Generated API cleanup preserves maintained pages and rolls back failures."""

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

API_TOOLS = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "docs/generate_apidoc.py")
)


def make_module_page(module: str) -> str:
    """Create a legacy generated page without the new ownership marker."""
    return (
        f"{module} module\n{'=' * (len(module) + 7)}\n\n"
        f".. automodule:: {module}\n"
        "   :members:\n   :show-inheritance:\n   :undoc-members:\n"
    )


@pytest.fixture
def docs_repo(tmp_path: Path) -> Path:
    """Create a checkout containing generated and maintained documentation."""
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    source = tmp_path / "docs/source"
    source.mkdir(parents=True)
    (tmp_path / ".gitignore").write_text(
        "pyvale.rst\npyvale.*.rst\n!docs/source/pyvale.unignored.rst\n"
    )
    (source / "pyvale.sensorsim.raster.rst").write_text(
        make_module_page("pyvale.sensorsim.raster")
    )
    (source / "pyvale.render.uvtools.rst").write_text("Maintained UV documentation\n")
    (source / "pyvale.manual.rst").write_text(
        make_module_page("pyvale.manual") + "\nKeep these hand-written notes.\n"
    )
    (source / "pyvale.unignored.rst").write_text(make_module_page("pyvale.unignored"))
    (source / "modules.rst").write_text("Maintained index\n")
    (source / "examples").mkdir()
    (source / "examples/examples_mooseherder.rst").write_text("Manual gallery\n")
    subprocess.run(
        ["git", "add", "-f", "docs/source/pyvale.render.uvtools.rst"],
        cwd=tmp_path,
        check=True,
    )
    return tmp_path


def test_cleanup_preserves_maintained_pages_and_removes_stale_modules(
    docs_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = docs_repo / "docs/source"
    protected = {
        path: path.read_bytes()
        for path in source.rglob("*.rst")
        if path.name != "pyvale.sensorsim.raster.rst"
    }
    original_run = subprocess.run

    def generate(command, **kwargs):
        if command[:3] != [sys.executable, "-m", "sphinx.ext.apidoc"]:
            return original_run(command, **kwargs)
        assert not (source / "pyvale.sensorsim.raster.rst").exists()
        staged = Path(command[command.index("--output-dir") + 1])
        staged.mkdir()
        (staged / "pyvale.rst").write_text(make_module_page("pyvale"))
        for module in [
            "pyvale.sensorsim",
            "pyvale.render.uvtools",
            "pyvale.manual",
            "pyvale.unignored",
            "pyvale.dic.dic2dcpp",
        ]:
            (staged / f"{module}.rst").write_text(make_module_page(module))
        (staged / "modules.rst").write_text("Generated index\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", generate)
    API_TOOLS["generate_api_docs"](docs_repo)
    assert not (source / "pyvale.sensorsim.raster.rst").exists()
    assert not (source / "pyvale.dic.dic2dcpp.rst").exists()
    assert (
        (source / "pyvale.sensorsim.rst")
        .read_text()
        .startswith(API_TOOLS["GENERATED_MARKER"])
    )
    for path, content in protected.items():
        assert path.read_bytes() == content
    assert not list((docs_repo / "docs").glob(".apidoc-*"))


@pytest.mark.parametrize("fail_during", ["generation", "publication"])
def test_failed_regeneration_restores_previous_pages(
    docs_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_during: str,
) -> None:
    source = docs_repo / "docs/source"
    original_pages = {path: path.read_bytes() for path in source.rglob("*.rst")}
    original_run = subprocess.run
    original_write_text = Path.write_text

    def generate(command, **kwargs):
        if command[:3] != [sys.executable, "-m", "sphinx.ext.apidoc"]:
            return original_run(command, **kwargs)
        assert not (source / "pyvale.sensorsim.raster.rst").exists()
        if fail_during == "generation":
            raise subprocess.CalledProcessError(1, command)
        staged = Path(command[command.index("--output-dir") + 1])
        staged.mkdir()
        (staged / "pyvale.rst").write_text(make_module_page("pyvale"))
        (staged / "pyvale.new.rst").write_text(make_module_page("pyvale.new"))
        return subprocess.CompletedProcess(command, 0)

    def write_text(path, *args, **kwargs):
        if path == source / "pyvale.new.rst":
            raise OSError("Cannot publish API page")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", generate)
    monkeypatch.setattr(Path, "write_text", write_text)
    with pytest.raises((subprocess.CalledProcessError, OSError)):
        API_TOOLS["generate_api_docs"](docs_repo)
    assert {path: path.read_bytes() for path in source.rglob("*.rst")} == original_pages
    assert not list((docs_repo / "docs").glob(".apidoc-*"))


def test_tracked_generated_template_is_protected(
    docs_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = docs_repo / "docs/source"
    page = source / "pyvale.render.uvtools.rst"
    page.write_text(make_module_page("pyvale.render.uvtools"))
    original = page.read_bytes()
    original_run = subprocess.run

    def fail_generation(command, **kwargs):
        if command[:3] == [sys.executable, "-m", "sphinx.ext.apidoc"]:
            assert page.read_bytes() == original
            raise subprocess.CalledProcessError(1, command)
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", fail_generation)
    with pytest.raises(subprocess.CalledProcessError):
        API_TOOLS["generate_api_docs"](docs_repo)
    assert page.read_bytes() == original


@pytest.mark.parametrize(
    "module, heading",
    [
        ("pyvale.dataio.meshtools", "dataio.meshtools"),
        ("pyvale.sensorsim.plotting_logs", "pyvale.sensorsim.plotting\\_logs module"),
    ],
)
def test_legacy_short_and_escaped_headings_are_recognised(
    tmp_path: Path,
    module: str,
    heading: str,
) -> None:
    page = tmp_path / f"{module}.rst"
    text = make_module_page(module)
    text = text.replace(f"{module} module", heading, 1)
    page.write_text(text)
    assert API_TOOLS["is_generated_page"](page)
    page.write_text(text + "\nManually maintained explanation.\n")
    assert not API_TOOLS["is_generated_page"](page)


def test_empty_successful_generation_also_restores_previous_pages(
    docs_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = docs_repo / "docs/source"
    original_pages = {path: path.read_bytes() for path in source.rglob("*.rst")}
    original_run = subprocess.run

    def empty_generation(command, **kwargs):
        if command[:3] == [sys.executable, "-m", "sphinx.ext.apidoc"]:
            return subprocess.CompletedProcess(command, 0)
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", empty_generation)
    with pytest.raises(RuntimeError, match="root pyvale.rst"):
        API_TOOLS["generate_api_docs"](docs_repo)
    assert {path: path.read_bytes() for path in source.rglob("*.rst")} == original_pages


def test_public_api_excludes_unfinished_and_internal_packages(docs_repo: Path) -> None:
    """The curated API is the root; public Render backend pages remain available."""
    package = docs_repo / "src/pyvale"
    packages = [
        "", "dataio", "render", "render/blender", "blender", "commoncpp", "valid"
    ]
    for name in packages:
        directory = package / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text('"""Example package."""\n')
    API_TOOLS["generate_api_docs"](docs_repo)
    source = docs_repo / "docs/source"
    assert (source / "pyvale.dataio.rst").is_file()
    assert (source / "pyvale.render.blender.rst").is_file()
    assert not (source / "pyvale.rst").exists()
    for name in ["blender", "commoncpp", "valid"]:
        assert not list(source.glob(f"pyvale.{name}*.rst"))
