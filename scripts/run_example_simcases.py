# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================
"""Regenerate the simulation outputs distributed as Pyvale package data.

The MOOSE and Gmsh input decks are copied into an isolated temporary directory
before execution. Only the requested Exodus outputs are copied back into the
package-data directory, so source decks and generated meshes remain unchanged.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
import shutil
import tempfile
import time

from pyvale.mooseherder import GmshRunner, MooseConfig, MooseRunner


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
"""Input-deck stems whose Exodus results are distributed with Pyvale."""

SIMCASES_DIR = Path(files("pyvale.data").joinpath("simulation", "simcases"))
EXODUS_DIR = Path(files("pyvale.data").joinpath("simulation", "exodus"))


def _copy_case_inputs(case_name: str, work_dir: Path) -> Path:
    """Copy a package-data case deck and its optional Gmsh source to ``work_dir``."""
    input_path = SIMCASES_DIR / f"{case_name}.i"
    if not input_path.is_file():
        raise FileNotFoundError(f"MOOSE input deck does not exist: {input_path}")

    shutil.copy2(input_path, work_dir / input_path.name)
    geometry_path = SIMCASES_DIR / f"{case_name}.geo"
    if geometry_path.is_file():
        shutil.copy2(geometry_path, work_dir / geometry_path.name)

    return work_dir / input_path.name


def _run_case(
    case_name: str,
    work_dir: Path,
    moose_runner: MooseRunner,
    gmsh_runner: GmshRunner,
) -> tuple[Path, float, float]:
    """Generate one case in ``work_dir`` and return its output and run times."""
    input_path = _copy_case_inputs(case_name, work_dir)
    geometry_path = input_path.with_suffix(".geo")

    gmsh_run_time = 0.0
    if geometry_path.is_file():
        gmsh_start = time.perf_counter()
        gmsh_runner.run(geometry_path)
        gmsh_run_time = time.perf_counter() - gmsh_start

    moose_start = time.perf_counter()
    moose_runner.run(input_path)
    moose_run_time = time.perf_counter() - moose_start
    output_path = moose_runner.get_output_path()
    if output_path is None or not output_path.is_file():
        raise RuntimeError(f"MOOSE did not produce an Exodus output for {case_name}.")

    return output_path, gmsh_run_time, moose_run_time


def regenerate_cases(
    case_names: Sequence[str],
    *,
    moose_main_path: Path,
    moose_app_path: Path,
    moose_app_name: str,
    gmsh_path: Path | None,
    n_threads: int,
) -> None:
    """Regenerate package Exodus data from copied input decks.

    The temporary workspace is removed whether a backend succeeds or fails.
    """
    unknown_cases = set(case_names).difference(PACKAGE_DATA_CASES)
    if unknown_cases:
        names = ", ".join(sorted(unknown_cases))
        raise ValueError(f"Cases do not have packaged Exodus outputs: {names}")

    config = MooseConfig(
        {
            "main_path": moose_main_path,
            "app_path": moose_app_path,
            "app_name": moose_app_name,
        },
    )
    moose_runner = MooseRunner(config)
    moose_runner.set_run_opts(n_tasks=1, n_threads=n_threads, redirect_out=False)
    gmsh_runner = GmshRunner(gmsh_path)

    with tempfile.TemporaryDirectory(prefix="pyvale-simcases-") as temp_dir:
        root_dir = Path(temp_dir)
        for case_name in case_names:
            print(f"Regenerating {case_name}.")
            case_dir = root_dir / case_name
            case_dir.mkdir()
            output_path, gmsh_time, moose_time = _run_case(
                case_name,
                case_dir,
                moose_runner,
                gmsh_runner,
            )
            destination = EXODUS_DIR / output_path.name
            shutil.copy2(output_path, destination)
            print(
                f"  copied {output_path.name}; Gmsh: {gmsh_time:.2f}s; "
                f"MOOSE: {moose_time:.2f}s",
            )


def _parse_args() -> argparse.Namespace:
    """Parse command-line configuration for the local MOOSE toolchain."""
    home_dir = Path.home()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=PACKAGE_DATA_CASES,
        default=PACKAGE_DATA_CASES,
        help="Package-data cases to regenerate.",
    )
    parser.add_argument("--moose-main-path", type=Path, default=home_dir / "moose")
    parser.add_argument("--moose-app-path", type=Path, default=home_dir / "proteus")
    parser.add_argument("--moose-app-name", default="proteus-opt")
    parser.add_argument("--gmsh-path", type=Path, default=home_dir / "gmsh/bin/gmsh")
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    """Regenerate the selected package-data simulation results."""
    args = _parse_args()
    regenerate_cases(
        args.cases,
        moose_main_path=args.moose_main_path,
        moose_app_path=args.moose_app_path,
        moose_app_name=args.moose_app_name,
        gmsh_path=args.gmsh_path,
        n_threads=args.threads,
    )


if __name__ == "__main__":
    main()
