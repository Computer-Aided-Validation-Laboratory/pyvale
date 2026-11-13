"""Native reference generation preserves authored pages and unique symbols."""

from pathlib import Path
import os
import runpy
import shutil
import subprocess
import sys

import pytest


TOOLS = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "docs/generate_native_apidoc.py")
)


def make_legacy_page(header: str) -> str:
    """Return the exact bare header template produced by the former script."""
    return (
        f"{header}\n{'=' * len(header)}\n\n.. doxygenfile:: {header}\n"
        "   :project: pyvale\n   :path: ../src/pyvale/dic/cpp\n"
    )


@pytest.fixture
def native_repo(tmp_path: Path) -> Path:
    """Provide two headers sharing a namespace and a separate strain header."""
    headers = tmp_path / "src/pyvale/dic/cpp"
    headers.mkdir(parents=True)
    (headers / "stereomatching.hpp").write_text(
        "namespace stereo {\n/** Match a stereo pair. */\nvoid matching();\n}\n"
    )
    (headers / "stereoutil.hpp").write_text(
        "namespace stereo {\n/** Compute stereo geometry. */\nvoid geometry();\n}\n"
    )
    strain = tmp_path / "src/pyvale/strain/cpp"
    strain.mkdir(parents=True)
    (strain / "strain.hpp").write_text(
        "namespace strain {\n/** Calculate strain. */\nvoid engine();\n}\n"
    )
    output = tmp_path / "docs/source/cpp/dic"
    output.mkdir(parents=True)
    (output / "diccoursefine.rst").write_text(make_legacy_page("diccoursefine.hpp"))
    (output.parent / "cpp_dic.rst").write_text(
        "dic\n===\n\n.. toctree::\n   :caption: C++ Source Files\n"
        "   :maxdepth: 1\n\n   dic/diccoursefine.rst\n"
    )
    return tmp_path


def test_native_generation_migrates_legacy_pages_and_preserves_manual_content(
    native_repo: Path,
) -> None:
    """Stale templates disappear while current headers and manual pages survive."""
    output = native_repo / "docs/source/cpp"
    headers = native_repo / "src/pyvale/dic/cpp"
    (headers / "manual.hpp").write_text("struct Manual {};\n")
    manual = output / "dic/manual.rst"
    manual.write_text("Maintained native notes\n=======================\n")
    original_manual = manual.read_bytes()
    (headers / "linked.hpp").write_text("struct Linked {};\n")
    linked = output / "dic/linked.rst"
    linked.symlink_to(manual.name)

    TOOLS["generate_native_api_docs"](native_repo)
    assert not (output / "dic/diccoursefine.rst").exists()
    index = (output / "cpp_dic.rst").read_text()
    assert "diccoursefine" not in index
    assert "dic/stereomatching" in index and "dic/stereoutil" in index
    assert "dic/manual" in index
    assert manual.read_bytes() == original_manual and linked.is_symlink()
    matching = (output / "dic/stereomatching.rst").read_text()
    assert matching.startswith(TOOLS["GENERATED_MARKER"])
    assert ".. doxygenfunction:: stereo::matching" in matching
    assert ":no-link:" not in matching
    assert ":path:" not in matching
    # Removing the marker from new output makes it manually maintained.
    utility = output / "dic/stereoutil.rst"
    utility.write_text(
        utility.read_text().replace(TOOLS["GENERATED_MARKER"] + "\n", "")
    )
    assert not TOOLS["is_generated_native_page"](utility)
    original_utility = utility.read_bytes()
    TOOLS["generate_native_api_docs"](native_repo)
    assert utility.read_bytes() == original_utility


def test_native_generation_restores_files_after_publication_failure(
    native_repo: Path,
) -> None:
    """A blocked output directory does not leave partial migrations behind."""
    output = native_repo / "docs/source/cpp"
    (output / "strain").write_text("This file blocks the output directory.\n")
    original = {page: page.read_bytes() for page in output.rglob("*.rst")}
    with pytest.raises(FileExistsError):
        TOOLS["generate_native_api_docs"](native_repo)
    assert {page: page.read_bytes() for page in output.rglob("*.rst")} == original


def test_legacy_page_with_authored_notes_is_not_owned(native_repo: Path) -> None:
    """Appended prose protects a legacy-looking page from regeneration."""
    page = native_repo / "docs/source/cpp/dic/diccoursefine.rst"
    page.write_text(page.read_text() + "\nKeep this explanation.\n")
    assert not TOOLS["is_generated_native_page"](page)


@pytest.mark.skipif(shutil.which("doxygen") is None, reason="Doxygen is required")
def test_native_reference_build_has_unique_stereo_symbols(native_repo: Path) -> None:
    """Doxygen/Breathe builds both stereo entry points with warnings as errors."""
    source = native_repo / "docs/source"
    xml = native_repo / "doxygen"
    doxyfile = source / "Doxyfile"
    doxyfile.write_text(
        f'INPUT = "{native_repo / "src/pyvale"}"\n'
        f'OUTPUT_DIRECTORY = "{xml}"\n'
        "RECURSIVE = YES\nFILE_PATTERNS = *.hpp\nEXTRACT_ALL = YES\n"
        "GENERATE_HTML = NO\nGENERATE_LATEX = NO\nGENERATE_XML = YES\n"
        "QUIET = YES\nWARN_IF_UNDOCUMENTED = NO\n"
    )
    # Exercise the actual shell entry point with a relative PYTHON path. It
    # must keep selecting that environment after running Doxygen elsewhere.
    project_root = Path(__file__).resolve().parents[2]
    for name in ["apidoc.sh", "generate_apidoc.py", "generate_native_apidoc.py"]:
        shutil.copyfile(project_root / "docs" / name, native_repo / "docs" / name)
    for package in ["", "dic", "strain"]:
        (native_repo / "src/pyvale" / package / "__init__.py").write_text("")
    subprocess.run(["git", "init", "--quiet", str(native_repo)], check=True)
    environment = os.environ.copy()
    environment["PYTHON"] = os.path.relpath(sys.executable, native_repo)
    subprocess.run(
        ["bash", "docs/apidoc.sh"],
        cwd=native_repo,
        env=environment,
        check=True,
        capture_output=True,
    )
    (source / "conf.py").write_text(
        "extensions = ['breathe']\n"
        f"breathe_projects = {{'pyvale': {str(xml / 'xml')!r}}}\n"
    )
    (source / "index.rst").write_text(
        "Native API\n==========\n\n.. toctree::\n\n"
        "   cpp/cpp_dic\n   cpp/cpp_strain\n   pyvale.dic\n   pyvale.strain\n"
    )
    output = native_repo / "html"
    completed = subprocess.run(
        [sys.executable, "-m", "sphinx", "-W", "-b", "html", str(source), str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        completed.stdout[-4000:] + completed.stderr[-4000:]
    )
    assert "stereo::matching" in (output / "cpp/dic/stereomatching.html").read_text()
    assert "stereo::geometry" in (output / "cpp/dic/stereoutil.html").read_text()
