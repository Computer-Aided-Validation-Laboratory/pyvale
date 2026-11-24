#!/usr/bin/env python3
# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Run validation examples sequentially and retain figures in pyvale-output.

Run from the project root with the project Python environment. Use --data-dir
to select a local pyvale-data/Data checkout; otherwise normal data lookup and
the existing PYVALE_DATA environment variable apply.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, help="Local pyvale-data/Data directory"
    )
    arguments = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    examples_directory = repo_root / "src/pyvale/examples/validation"
    examples = sorted(examples_directory.glob("ex[0-9][0-9]*_valid_*.py"))
    if not examples:
        parser.error("No validation examples found in this checkout.")
    output_root = repo_root / "pyvale-output"
    log_directory = output_root / "valid_logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["PYVISTA_OFF_SCREEN"] = "true"
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(repo_root / "src"), environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    if arguments.data_dir is not None:
        data_directory = arguments.data_dir.expanduser().resolve()
        if not data_directory.is_dir():
            parser.error(f"Data directory does not exist: {data_directory}")
        environment["PYVALE_DATA"] = str(data_directory)

    failed: list[str] = []
    for index, example in enumerate(examples, start=1):
        print(f"[{index}/{len(examples)}] {example.name}", flush=True)
        start = perf_counter()
        output_directory = output_root / (
            "valid_" + example.stem.replace("_valid_", "_", 1)
        )
        log_path = log_directory / f"{example.stem}.log"
        with log_path.open("w") as log_stream:
            completed = subprocess.run(
                (sys.executable, str(example)),
                cwd=repo_root,
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            failed.append(example.name)
            print(f"  FAIL ({completed.returncode}); see {log_path}", flush=True)
        else:
            location = output_directory if output_directory.is_dir() else log_path
            print(
                f"  PASS ({perf_counter() - start:.1f}s): {location}",
                flush=True,
            )

    print(f"\n{len(examples) - len(failed)} passed, {len(failed)} failed")
    print(f"Outputs: {output_root}/valid_ex*")
    print(f"Logs: {log_directory}")
    if failed:
        print("Failed examples: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
