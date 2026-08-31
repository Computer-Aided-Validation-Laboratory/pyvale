#!/usr/bin/env python3
# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Run all pyvale gallery examples.

Discovers example scripts in the sub-directories of ``src/pyvale/examples``
listed in ``EXAMPLE_DIRS`` and executes each one in an isolated temporary
working directory. Each working directory links its standard
``pyvale-output`` path to the repository-level output directory, matching the
location used when examples are run standalone from the repository root. A
summary is printed at the end and the exit code is non-zero if any example
failed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "src" / "pyvale" / "examples"
OUTPUT_ROOT = REPO_ROOT / "pyvale-output"

#: Sub-directories of ``src/pyvale/examples`` that contain examples to run.
EXAMPLE_DIRS = [
#    "basicsensorsim",
#   "extsensorsim",
#    "dic",
#    "mooseherder",
    "render3d",
    "renderuvs",
]

#: Examples that cannot run unattended because they open interactive GUI
#: windows or require unbundled local input files.
EXCLUDED_EXAMPLES = {
    "ex01_region_of_interest.py",
    "ex05_dic_challenge.py",
    "ex06_hrdic.py",
    "ex08_calibration.py",
    "ex09_stereo.py",
    "ex10_stereo_platehole.py",
    "ex11_dic_chal.py",
}

#: Maximum wall-clock time allowed per example.
TIMEOUT_S = 600.0

#: Non-script sibling files copied next to each example before it runs
#: (ROI definitions, calibration data, solver configuration files).
SUPPORT_FILE_GLOBS = ("*.txt", "*.yaml", "*.json")


def copy_support_files(script: Path, work_dir: Path) -> None:
    """Copy an example's sibling support files into the working directory."""
    for pattern in SUPPORT_FILE_GLOBS:
        for support_file in script.parent.glob(pattern):
            shutil.copy2(support_file, work_dir / support_file.name)


def link_output_directory(work_dir: Path) -> None:
    """Link an example working directory to the standard output directory."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (work_dir / "pyvale-output").symlink_to(
        OUTPUT_ROOT,
        target_is_directory=True,
    )


def find_examples() -> list[Path]:
    """Return the runnable example scripts in a stable run order."""
    examples: list[Path] = []
    for dir_name in EXAMPLE_DIRS:
        directory = EXAMPLES_ROOT / dir_name
        if not directory.is_dir():
            print(f"WARNING: example directory not found: {directory}")
            continue
        for script in sorted(directory.glob("ex*.py")):
            if script.name not in EXCLUDED_EXAMPLES:
                examples.append(script)
    return examples


def run_example(script: Path, work_dir: Path) -> tuple[bool, float, str]:
    """Run one example in ``work_dir``.

    Parameters
    ----------
    script : pathlib.Path
        The example script to execute.
    work_dir : pathlib.Path
        Temporary working directory the example runs in.

    Returns
    -------
    tuple[bool, float, str]
        Success flag, elapsed seconds, and a short status message.
    """
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["PYVISTA_OFF_SCREEN"] = "true"
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPO_ROOT / "src"), environment.get("PYTHONPATH", "")),
    ).rstrip(os.pathsep)

    start_time = time.perf_counter()
    try:
        completed = subprocess.run(
            (sys.executable, str(script)),
            cwd=work_dir,
            env=environment,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        message = f"timed out after {TIMEOUT_S:.0f} s"
        return False, time.perf_counter() - start_time, message

    elapsed = time.perf_counter() - start_time
    if completed.returncode != 0:
        message = f"exit code {completed.returncode}"
        print("-" * 70)
        print(f"FAILED: {script.name} ({message})")
        if completed.stdout:
            print("stdout:\n" + completed.stdout)
        if completed.stderr:
            print("stderr:\n" + completed.stderr)
        print("-" * 70)
        return False, elapsed, message
    return True, elapsed, "ok"


def main() -> None:
    """Run every discovered example and print a pass/fail summary."""
    examples = find_examples()
    print(f"Running {len(examples)} examples from {len(EXAMPLE_DIRS)} "
          f"directories (timeout {TIMEOUT_S:.0f} s each).\n")

    session_dir = Path(tempfile.mkdtemp(prefix="pyvale-examples-"))
    passed: list[str] = []
    failed: list[tuple[str, str]] = []
    work_dirs: dict[str, Path] = {}

    for number, script in enumerate(examples, start=1):
        label = f"{script.parent.name}/{script.name}"
        print(f"[{number:>2}/{len(examples)}] RUN   {label}", flush=True)
        dir_name = script.parent.name
        if dir_name not in work_dirs:
            work_dirs[dir_name] = session_dir / dir_name
            work_dirs[dir_name].mkdir(parents=True)
            link_output_directory(work_dirs[dir_name])
        work_dir = work_dirs[dir_name]
        copy_support_files(script, work_dir)

        success, elapsed, message = run_example(script, work_dir)
        minutes = elapsed / 60.0
        if success:
            passed.append(label)
            print(f"      PASS  {label} ({minutes:.1f} min)")
        else:
            failed.append((label, message))
            print(f"      FAIL  {label} ({message})")

    skipped = sorted(
        f"{directory.name}/{name}"
        for directory in (EXAMPLES_ROOT / name for name in EXAMPLE_DIRS)
        if directory.is_dir()
        for name in EXCLUDED_EXAMPLES
        if (directory / name).is_file()
    )

    print("\n" + "=" * 70)
    print(f"EXAMPLE RUN COMPLETE: {len(passed)} passed, {len(failed)} failed, "
          f"{len(skipped)} excluded")
    if skipped:
        print("\nExcluded (interactive GUI or unbundled input):")
        for label in skipped:
            print(f"  SKIP  {label}")
    if failed:
        print("\nFailed examples:")
        for label, message in failed:
            print(f"  FAIL  {label} ({message})")
    print("=" * 70)
    print(f"Example output kept for inspection: {OUTPUT_ROOT}")
    print(f"Session support files kept for inspection: {session_dir}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
