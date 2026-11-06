"""Deliberately regenerate reviewed package-data Exodus baselines.

Run only after reviewing numerical changes. Ordinary tests never regenerate
these files. All meshes, inputs and logs are produced in a temporary directory.
"""

import argparse
import shutil
import tempfile
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

from run_simcase import run_case

PACKAGE_DATA_CASES = (
    "case00_TET4",
    "case00_TET10",
    "case00_TET14",
    "case00_HEX8",
    "case00_HEX20",
    "case00_HEX27",
    "case16",
    "case16_d",
    "case17",
    "case18",
    "case18_d",
    "case26",
)
EXODUS_DIR = Path(files("pyvale.data").joinpath("simulation", "exodus"))


def regenerate_cases(
    case_names: Sequence[str],
    *,
    moose_executable: Path | None = None,
    gmsh_executable: Path | None = None,
    n_threads: int = 1,
) -> None:
    """Replace selected reviewed baselines, preserving their public filenames."""
    unknown = set(case_names) - set(PACKAGE_DATA_CASES)
    if unknown:
        raise ValueError(f"Unknown package-data cases: {unknown}")
    with tempfile.TemporaryDirectory(prefix="pyvale-simcases-") as directory:
        for name in case_names:
            result = run_case(
                name,
                Path(directory) / name,
                moose_executable=moose_executable,
                gmsh_executable=gmsh_executable,
                n_threads=n_threads,
            )
            target = EXODUS_DIR / f"{name}_out.e"
            shutil.copy2(result.output, target)
            print(f"Copied reviewed output: {target}")


def main() -> None:
    """Regenerate only explicitly selected package-data baselines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=PACKAGE_DATA_CASES, required=True)
    parser.add_argument("--moose-executable", type=Path)
    parser.add_argument("--gmsh-executable", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    regenerate_cases(
        args.cases,
        moose_executable=args.moose_executable,
        gmsh_executable=args.gmsh_executable,
        n_threads=args.threads,
    )


if __name__ == "__main__":
    main()
