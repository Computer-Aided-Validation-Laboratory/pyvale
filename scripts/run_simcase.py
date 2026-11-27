# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Run a packaged case with Rabbit/Gmsh in an isolated output directory."""

import argparse
from importlib.resources import files
from pathlib import Path

from pyvale.mooseherder import GmshRunner, MooseRunner, RunResult

SIMCASES_DIR = Path(files("pyvale.data").joinpath("simulation", "simcases"))


def run_case(
    case_name: str,
    output_dir: Path,
    *,
    moose_exe_file: Path | None = None,
    gmsh_exe_file: Path | None = None,
    n_tasks: int | None = None,
    n_threads: int = 1,
) -> RunResult:
    """Generate a packaged case's mesh if required and solve its selected deck.

    Parameters
    ----------
    case_name : str
        Packaged case stem, for example case17 or case00_HEX8. MPI entry suffixes
        are selected through n_tasks rather than changing the physical model.
    output_dir : Path
        Case output root; mesh and solve steps use separate new subdirectories.
    moose_exe_file : Path | None, optional
        Custom MOOSE application. None selects packaged Rabbit.
    gmsh_exe_file : Path | None, optional
        Custom Gmsh executable. None selects Python Gmsh.
    n_tasks : int | None, optional
        None or 1 runs serially; greater counts select an MPI deck and launcher.
    n_threads : int, optional
        Positive threads per solver process. Defaults to 1.

    Returns
    -------
    RunResult
        Completed solver output, working directory, invocation and log.

    Raises
    ------
    FileNotFoundError
        The requested packaged input deck does not exist.
    RuntimeError
        A required backend or launcher is unavailable, or meshing/solving fails.

    Notes
    -----
    The model's native units are retained. Mesh variants such as case16_d reuse
    case16.geo when no separate geometry exists. Packaged sources are not edited.
    """
    mpi = n_tasks is not None and n_tasks > 1
    base_name = case_name.removesuffix("_mpi")
    deck = SIMCASES_DIR / f"{base_name}{'_mpi' if mpi else ''}.i"
    if not deck.is_file():
        raise FileNotFoundError(deck)
    geometry = SIMCASES_DIR / f"{base_name}.geo"
    if not geometry.is_file() and base_name.endswith(("_d", "_u")):
        geometry = SIMCASES_DIR / f"{base_name[:-2]}.geo"
    input_params = {}
    if geometry.is_file():
        mesh = GmshRunner(gmsh_exe_file=gmsh_exe_file).run(
            geometry,
            output_dir=output_dir / "mesh",
        )
        text = (SIMCASES_DIR / f"{base_name}.i").read_text(encoding="utf-8")
        mesh_key = "mesh_file" if "mesh_file =" in text else "Mesh/file"
        input_params[mesh_key] = str(mesh.output_file)
    return MooseRunner(
        moose_exe_file=moose_exe_file,
        n_tasks=n_tasks,
        n_threads=n_threads,
    ).run(deck, input_params=input_params, output_dir=output_dir / "solve")


def main() -> None:
    """Run one case without changing its packaged source files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", help="For example case18 or case00_HEX8")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "pyvale-output")
    parser.add_argument("--moose-exe-file", type=Path)
    parser.add_argument("--gmsh-exe-file", type=Path)
    parser.add_argument("--tasks", type=int, default=None)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    result = run_case(
        args.case,
        args.output_dir / args.case,
        moose_exe_file=args.moose_exe_file,
        gmsh_exe_file=args.gmsh_exe_file,
        n_tasks=args.tasks,
        n_threads=args.threads,
    )
    print(result.output_file)


if __name__ == "__main__":
    main()
