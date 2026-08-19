"""Command-line worker used by Slurm and local-array workflow execution."""

import argparse
from pathlib import Path

from .executor import _run_persisted_case


def main() -> None:
    """Parse one case request and run its persisted workflow definition."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    arguments = parser.parse_args()
    _run_persisted_case(arguments.run_dir, arguments.case_index)


if __name__ == "__main__":
    main()
