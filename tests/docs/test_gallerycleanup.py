"""Gallery cleanup removes obsolete output without deleting authored pages."""

from pathlib import Path
import runpy
import subprocess

import pytest


TOOLS = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "docs/gallerycleanup.py")
)


def write_generated_page(source: Path, group: str, name: str) -> Path:
    """Create a gallery page with Sphinx-Gallery's ownership and download markers."""
    page = source / "examples" / group / f"{name}.rst"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(
        f"{TOOLS['GENERATED_MARKER']}\n"
        f'.. "examples/{group}/{name}.py"\n\n'
        f"Example\n=======\n\n"
        f":download:`Source <{name}.py>`\n"
        f":download:`Notebook <{name}.ipynb>`\n"
        f":download:`Archive <{name}.zip>`\n"
        ":download:`Unrelated file <../../manual.rst>`\n",
        encoding="utf-8",
    )
    return page


@pytest.fixture
def gallery_repo(tmp_path: Path) -> Path:
    """Provide a real Git repository with ignored gallery output."""
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text(
        "docs/source/examples/*/*\n"
        "!docs/source/examples/current/unignored.rst\n",
        encoding="utf-8",
    )
    (tmp_path / "docs/source/examples/current").mkdir(parents=True)
    (tmp_path / "src/pyvale/examples/current").mkdir(parents=True)
    return tmp_path


def test_cleanup_preserves_current_and_maintained_files(gallery_repo: Path) -> None:
    """Tracked, manual, unignored and symlink pages survive a stale-page cleanup."""
    source = gallery_repo / "docs/source"
    examples = gallery_repo / "src/pyvale/examples/current"
    (examples / "ex_current.py").write_text("print('current')\n")
    current = write_generated_page(source, "current", "ex_current")
    tracked = write_generated_page(source, "current", "tracked")
    subprocess.run(
        ["git", "add", "-f", str(tracked)], cwd=gallery_repo, check=True
    )
    unignored = write_generated_page(source, "current", "unignored")
    manual = source / "examples/current/manual.rst"
    manual.write_text("Maintained gallery notes\n========================\n")
    symlink = source / "examples/current/link.rst"
    symlink.symlink_to(current.name)
    protected = {
        path: path.read_bytes() for path in [current, tracked, unignored, manual]
    }

    stale = write_generated_page(source, "current", "ex_removed")
    companions = [
        stale.with_name(stale.stem + suffix)
        for suffix in (".py", ".ipynb", ".zip", ".codeobj.json", ".py.md5")
    ]
    for path in companions:
        path.write_text("generated output\n")
    notes = stale.with_name("ex_removed_notes.txt")
    notes.write_text("Keep unrelated notes\n")
    outside = source / "manual.rst"
    outside.write_text("Keep this file\n")
    # A tracked companion is protected even when its owning RST is obsolete.
    subprocess.run(
        ["git", "add", "-f", str(companions[0])], cwd=gallery_repo, check=True
    )

    removed = TOOLS["clean_generated_galleries"](
        source,
        ["../../src/pyvale/examples/current"],
        ["examples/current"],
        ignore_pattern=r"(?:^|/)__init__\.py$",
    )
    assert stale in removed and not stale.exists()
    assert companions[0].is_file()
    assert all(not path.exists() for path in companions[1:])
    assert all(path.read_bytes() == content for path, content in protected.items())
    assert symlink.is_symlink() and notes.is_file() and outside.is_file()
    assert TOOLS["clean_generated_galleries"](
        source,
        ["../../src/pyvale/examples/current"],
        ["examples/current"],
        ignore_pattern=r"(?:^|/)__init__\.py$",
    ) == []


def test_cleanup_removes_retired_blender_pages_and_excluded_helpers(
    gallery_repo: Path,
) -> None:
    """A live Render Blender example survives while a retired gallery is removed."""
    source = gallery_repo / "docs/source"
    examples = gallery_repo / "src/pyvale/examples/current"
    (examples / "ex_blender.py").write_text("print('current backend')\n")
    (examples / "__init__.py").write_text("")
    current = write_generated_page(source, "current", "ex_blender")
    retired = write_generated_page(source, "blenderimagedef", "ex_old_blender")
    helper = write_generated_page(source, "current", "__init__")

    TOOLS["clean_generated_galleries"](
        source,
        ["../../src/pyvale/examples/current"],
        ["examples/current"],
        ignore_pattern=r"(?:^|/)__init__\.py$",
    )
    assert current.exists()
    assert not retired.exists() and not helper.exists()


def test_cleanup_rejects_mismatched_gallery_mapping(gallery_repo: Path) -> None:
    """An invalid configuration fails before any file is removed."""
    with pytest.raises(ValueError, match="one gallery directory"):
        TOOLS["clean_generated_galleries"](
            gallery_repo / "docs/source", ["current"], [], ignore_pattern=""
        )
