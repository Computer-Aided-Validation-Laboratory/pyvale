# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Shared helpers for documented-example smoke tests."""

from __future__ import annotations

import os
from collections.abc import Callable, Generator, Sequence
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
EXAMPLES_ROOT = SOURCE_ROOT / "pyvale" / "examples"
EXAMPLE_TESTS_ROOT = Path(__file__).resolve().parent


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register filtering for one documented example sub-module."""
    parser.addoption(
        "--example-module",
        metavar="MODULE",
        help="Run example tests whose script is in this sub-module.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Limit example smoke tests to the requested example sub-module."""
    requested_module = config.getoption("example_module")
    if requested_module is None:
        return

    module = requested_module.strip("/")
    requested_prefix = f"{module}/"
    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []

    for item in items:
        if not item.path.is_relative_to(EXAMPLE_TESTS_ROOT):
            selected.append(item)
            continue

        callspec = getattr(item, "callspec", None)
        relative_path = (
            callspec.params.get("example") if callspec is not None else None
        )
        module_marker = item.get_closest_marker("example_module")
        marker_module = (
            module_marker.args[0] if module_marker is not None else None
        )

        if (
            isinstance(relative_path, str)
            and relative_path.startswith(requested_prefix)
        ) or marker_module == module:
            selected.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.fixture
def run_example() -> Generator[Callable[..., Path], None, None]:
    """Run an example script in an isolated working directory.

    The gallery examples write to ``pyvale-output`` relative to their current
    directory.  Running them in a temporary directory prevents example smoke
    tests from changing the source tree and keeps independent tests isolated.
    """

    with tempfile.TemporaryDirectory(prefix="pyvale-example-") as directory:
        work_dir = Path(directory)

        def _run(
            relative_path: str,
            expected_outputs: Sequence[str] = (),
            support_files: Sequence[str] = (),
            timeout: float = 180.0,
        ) -> Path:
            example_path = EXAMPLES_ROOT / relative_path
            assert example_path.is_file(), (
                f"Example does not exist: {example_path}"
            )

            for support_file in support_files:
                source_file = example_path.parent / support_file
                target_file = work_dir / support_file
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, target_file)

            environment = os.environ.copy()
            environment["MPLBACKEND"] = "Agg"
            environment["PYVISTA_OFF_SCREEN"] = "true"
            environment["QT_QPA_PLATFORM"] = "offscreen"
            environment["PYTHONPATH"] = os.pathsep.join(
                (str(SOURCE_ROOT), environment.get("PYTHONPATH", "")),
            ).rstrip(os.pathsep)

            completed = subprocess.run(
                (sys.executable, str(example_path)),
                cwd=work_dir,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

            assert completed.returncode == 0, (
                f"Example failed: {relative_path}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

            for output in expected_outputs:
                assert (work_dir / output).exists(), (
                    f"Example did not create expected output: {output}"
                )

            return work_dir

        yield _run
