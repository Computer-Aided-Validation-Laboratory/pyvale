"""Validate hand-written image references before generating gallery pages."""

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs/source"
EXAMPLE_GROUPS = (
    "basicsensorsim", "dic", "render3d", "renderuvs", "extsensorsim", "mooseherder"
)


def test_gallery_images_use_existing_source_root_assets() -> None:
    """Gallery narrative must resolve independently of its generated directory."""
    references = 0
    for group in EXAMPLE_GROUPS:
        for path in (REPO_ROOT / "src/pyvale/examples" / group).glob("ex*.py"):
            for target in re.findall(
                r"^\s*#.*?\.\. (?:image|figure)::\s+(\S+)",
                path.read_text(encoding="utf-8"),
                flags=re.MULTILINE,
            ):
                if target.startswith(("https://", "http://")):
                    continue
                references += 1
                assert target.startswith("/_static/"), (path, target)
                assert (DOCS_ROOT / target.lstrip("/")).is_file(), (path, target)
    assert references > 0


def test_stereo_theory_has_existing_light_and_dark_diagrams() -> None:
    """Both Furo theme variants must resolve to real, separate assets."""
    source = DOCS_ROOT / "guide_theory/guide_theory_dic_stereo.rst"
    diagrams = re.findall(
        r"\.\. image:: (\S+)\n\s+:class: (only-light|only-dark)",
        source.read_text(encoding="utf-8"),
    )
    assert {theme for _, theme in diagrams} == {"only-light", "only-dark"}
    assert len({target for target, _ in diagrams}) == 2
    for target, _ in diagrams:
        assert target.startswith("/_static/")
        assert (DOCS_ROOT / target.lstrip("/")).is_file()
