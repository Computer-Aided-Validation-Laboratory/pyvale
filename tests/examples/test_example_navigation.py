# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Regression tests for the hand-written example gallery navigation."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "src" / "pyvale" / "examples"
DOCS_EXAMPLES_ROOT = REPO_ROOT / "docs" / "source" / "examples"

GROUPS = (
    ("examples_basics_sensorsim", "basicsensorsim"),
    ("examples_dic", "dic"),
    ("examples_render3d", "render3d"),
    ("examples_renderuvs", "renderuvs"),
    ("examples_ext_sensorsim", "extsensorsim"),
    ("examples_mooseherder", "mooseherder"),
    ("examples_render2d", "render2d"),
)


def get_toctree_entries(path: Path, prefix: str) -> tuple[str, ...]:
    """Return ordered, extension-free toctree entries with one prefix."""
    entries: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if entry.startswith(prefix):
            entries.append(entry)
    return tuple(entries)


def test_example_groups_are_complete_and_ordered() -> None:
    """Every public example appears in its explicit filename order."""
    for group_name, source_name in GROUPS:
        group_path = DOCS_EXAMPLES_ROOT / f"{group_name}.rst"
        documented = get_toctree_entries(group_path, f"{source_name}/ex")
        source = tuple(
            f"{source_name}/{path.stem}"
            for path in sorted((EXAMPLES_ROOT / source_name).glob("ex*.py"))
        )
        assert documented == source


def test_example_group_navigation_order() -> None:
    """The examples landing page retains the intended module progression."""
    landing_page = DOCS_EXAMPLES_ROOT / "examples.rst"
    documented = get_toctree_entries(landing_page, "examples_")
    expected = tuple(group_name for group_name, _ in GROUPS)
    assert documented == expected


def test_development_example_families_are_not_public() -> None:
    """Supporting and in-development modules stay out of public navigation."""
    navigation = "\n".join(
        path.read_text(encoding="utf-8")
        for path in DOCS_EXAMPLES_ROOT.glob("examples*.rst")
    )
    for source_name in ("genanalyticdata", "valid", "visualisation"):
        assert source_name not in navigation


def test_gitignore_preserves_manual_navigation_only() -> None:
    """Manual layouts are trackable while generated gallery pages are ignored."""
    for group_name in ("examples", *(name for name, _ in GROUPS)):
        manual_path = DOCS_EXAMPLES_ROOT / f"{group_name}.rst"
        completed = subprocess.run(
            ("git", "check-ignore", "--quiet", str(manual_path)),
            cwd=REPO_ROOT,
            check=False,
        )
        assert completed.returncode == 1

    generated_path = DOCS_EXAMPLES_ROOT / "render3d" / "example.rst"
    completed = subprocess.run(
        ("git", "check-ignore", "--quiet", str(generated_path)),
        cwd=REPO_ROOT,
        check=False,
    )
    assert completed.returncode == 0
