"""Exercise the DIC user guide's code against the installed public APIs."""

import ast
from dataclasses import dataclass
import inspect
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import yaml

import pyvale.calib as calib
import pyvale.dic as dic
import pyvale.strain as strain


GUIDE_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs/source/guide_user/guide_dic.rst"
)


@dataclass(slots=True)
class GuideSnippet:
    language: str
    name: str | None
    code: str


def extract_guide_snippets() -> list[GuideSnippet]:
    """Read native RST code blocks, excluding directive options."""
    lines = GUIDE_PATH.read_text(encoding="utf-8").splitlines()
    snippets: list[GuideSnippet] = []
    for index, line in enumerate(lines):
        if not line.lstrip().startswith(".. code-block::"):
            continue
        indentation = len(line) - len(line.lstrip())
        language = line.split("::", 1)[1].strip().lower()
        name = None
        start = index + 1
        while start < len(lines) and (
            not lines[start].strip() or lines[start].lstrip().startswith(":")
        ):
            if lines[start].lstrip().startswith(":name:"):
                name = lines[start].split(":name:", 1)[1].strip()
            start += 1
        end = start
        while end < len(lines) and (
            not lines[end].strip()
            or len(lines[end]) - len(lines[end].lstrip()) > indentation
        ):
            end += 1
        snippets.append(
            GuideSnippet(language, name, textwrap.dedent("\n".join(lines[start:end])))
        )
    return snippets


def test_dic_guide_snippets_parse_and_match_signatures() -> None:
    """Reject syntax errors, placeholder arguments and obsolete API keywords."""
    functions = {
        ("dic", "calculate_2d"): dic.calculate_2d,
        ("dic", "calculate_3d"): dic.calculate_3d,
        ("dic", "import_2d"): dic.import_2d,
        ("dic", "import_3d"): dic.import_3d,
        ("strain", "calculate_2d"): strain.calculate_2d,
        ("strain", "import_2d"): strain.import_2d,
        ("calib", "loadtxt"): calib.loadtxt,
    }
    for snippet in extract_guide_snippets():
        if snippet.language == "yaml":
            shapes = yaml.safe_load(snippet.code)
            assert len(shapes) == 5
            assert all("type" in shape and "add" in shape for shape in shapes)
            continue
        tree = ast.parse(snippet.code)
        assert not any(
            isinstance(node, ast.Constant) and node.value is Ellipsis
            for node in ast.walk(tree)
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if not isinstance(node.func.value, ast.Name):
                continue
            key = (node.func.value.id, node.func.attr)
            if key in functions:
                inspect.signature(functions[key]).bind(
                    *(None for _ in node.args),
                    **{keyword.arg: None for keyword in node.keywords},
                )


@pytest.mark.data
@pytest.mark.parametrize(
    "snippet_name", ["dic-guide-complete-2d", "dic-guide-complete-stereo"]
)
def test_dic_guide_complete_workflow(tmp_path: Path, snippet_name: str) -> None:
    """Copy complete examples into fresh directories and execute real workflows."""
    snippets = extract_guide_snippets()
    primary = next(snippet for snippet in snippets if snippet.name == snippet_name)
    code = primary.code
    if snippet_name == "dic-guide-complete-2d":
        # Execute the non-interactive ROI alternatives before the full example,
        # in document order, so each later snippet uses the intended state.
        primary_index = snippets.index(primary)
        roi_code = "\n".join(
            snippet.code
            for snippet in snippets[:primary_index]
            if "roi.rect_boundary" in snippet.code or "roi_mask=arr" in snippet.code
        )
        code = roi_code + "\n" + primary.code
        # Plotting, strain and parameter examples use the complete 2D output.
        # The interactive GUI is validated statically above.
        for snippet in snippets[primary_index + 1:]:
            if snippet.language != "python" or snippet.name is not None:
                continue
            if any(
                marker in snippet.code
                for marker in (
                    "import matplotlib.pyplot",
                    "strain.calculate_2d",
                    "incremental_update=",
                    "max_displacement=512",
                    "fft_filter=True",
                    "for def_img in def_imgs",
                    "num_threads=4",
                )
            ):
                code += "\n" + snippet.code
        code += "\nassert np.isfinite(dic_results.u_px).sum() > 100\n"
        code += "assert np.isfinite(strain_data.eps_xx).sum() > 100\n"
        code += "print('MEDIAN_DISPLACEMENT', np.nanmedian(dic_results.u_px))\n"
    else:
        code += "\nimport numpy as np\n"
        code += "assert np.isfinite(stereo_results.stereo.x_mm).sum() > 100\n"
        code += "assert np.isfinite(stereo_results.stereo.u_mm).sum() > 100\n"
    script = tmp_path / "copied_guide.py"
    script.write_text(code, encoding="utf-8")
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, (
        f"{snippet_name}\n{completed.stdout[-8000:]}\n{completed.stderr[-8000:]}"
    )
    output = "dic_2d_results" if snippet_name.endswith("2d") else "dic_stereo_results"
    assert list((tmp_path / output).glob("dic_results_*.csv"))
