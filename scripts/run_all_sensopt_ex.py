#!/usr/bin/env python3
# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Run the sensor placement examples sequentially and save figures in pyvale-output."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_examples() -> int:
    """Run all OptSens examples with the current Python interpreter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, help="Local pyvale-data/Data directory"
    )
    arguments = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source_path = repo_root / "src"
    examples_path = source_path / "pyvale" / "examples" / "sensopt"
    examples = sorted(examples_path.glob("ex[0-9]*.py"))
    if not examples:
        parser.error("No sensor placement examples found in this checkout.")

    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["PYVISTA_OFF_SCREEN"] = "true"
    environment["QT_QPA_PLATFORM"] = "offscreen"
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join((str(source_path), python_path))
        if python_path
        else str(source_path)
    )
    if arguments.data_dir is not None:
        data_path = arguments.data_dir.expanduser().resolve()
        if not data_path.is_dir():
            parser.error(f"Data directory does not exist: {data_path}")
        environment["PYVALE_DATA"] = str(data_path)

    failed_examples: list[str] = []
    for index, example in enumerate(examples, start=1):
        print(f"[{index}/{len(examples)}] {example.name}", flush=True)
        completed = subprocess.run(
            (sys.executable, str(example)),
            cwd=repo_root,
            env=environment,
            check=False,
        )
        if completed.returncode:
            print(f"  Failed with exit code {completed.returncode}", flush=True)
            failed_examples.append(example.name)

    print(
        f"\n{len(examples) - len(failed_examples)} passed, "
        f"{len(failed_examples)} failed"
    )
    if failed_examples:
        print("Failed examples " + ", ".join(failed_examples))
    print(f"Outputs in {repo_root / 'pyvale-output'}")
    return 1 if failed_examples else 0


if __name__ == "__main__":
    sys.exit(run_examples())
