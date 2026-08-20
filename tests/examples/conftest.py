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
