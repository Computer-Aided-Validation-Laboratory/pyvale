# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Deliberately regenerate reviewed package-data Exodus baselines.

Run only after reviewing numerical changes. Ordinary tests never regenerate
these files. All meshes, input_files and logs are produced in a temporary directory.
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
    moose_exe_file: Path | None = None,
    gmsh_exe_file: Path | None = None,
    n_threads: int = 1,
) -> None:
    """Replace deliberately reviewed package-data Exodus baselines.

    Parameters
    ----------
    case_names : Sequence[str]
        Case stems selected from PACKAGE_DATA_CASES.
    moose_exe_file : Path | None, optional
        Custom MOOSE application path. None uses packaged Rabbit.
    gmsh_exe_file : Path | None, optional
        Custom Gmsh executable path. None uses Python Gmsh.
    n_threads : int, optional
        Positive thread count per solver process. Defaults to 1.

    Raises
    ------
    ValueError
        A requested case has no registered packaged baseline.
    RuntimeError
        Meshing or solving fails for a selected case.

    Notes
    -----
    Inputs, meshes and logs are produced in a temporary directory. Only requested
    Exodus baselines are copied into the package-data tree, retaining their public
    filenames. This overwrites existing baselines and is never part of tests.
    """
    unknown = set(case_names) - set(PACKAGE_DATA_CASES)
    if unknown:
        raise ValueError(f"Unknown package-data cases: {unknown}")
    with tempfile.TemporaryDirectory(prefix="pyvale-simcases-") as directory:
        for name in case_names:
            result = run_case(
                name,
                Path(directory) / name,
                moose_exe_file=moose_exe_file,
                gmsh_exe_file=gmsh_exe_file,
                n_threads=n_threads,
            )
            target = EXODUS_DIR / f"{name}_out.e"
            shutil.copy2(result.output_file, target)
            print(f"Copied reviewed output: {target}")


def main() -> None:
    """Regenerate only explicitly selected package-data baselines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=PACKAGE_DATA_CASES, required=True)
    parser.add_argument("--moose-exe-file", type=Path)
    parser.add_argument("--gmsh-exe-file", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    regenerate_cases(
        args.cases,
        moose_exe_file=args.moose_exe_file,
        gmsh_exe_file=args.gmsh_exe_file,
        n_threads=args.threads,
    )


if __name__ == "__main__":
    main()
