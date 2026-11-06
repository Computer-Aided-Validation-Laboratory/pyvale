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
    moose_executable: Path | None = None,
    gmsh_executable: Path | None = None,
    mpi: bool = False,
    n_tasks: int = 1,
    n_threads: int = 1,
) -> RunResult:
    """Generate a case mesh if needed, then run the matching SMP/MPI deck."""
    base_name = case_name.removesuffix("_mpi")
    deck = SIMCASES_DIR / f"{base_name}{'_mpi' if mpi else ''}.i"
    if not deck.is_file():
        raise FileNotFoundError(deck)
    geometry = SIMCASES_DIR / f"{base_name}.geo"
    if not geometry.is_file() and base_name.endswith(("_d", "_u")):
        geometry = SIMCASES_DIR / f"{base_name[:-2]}.geo"
    parameters = {}
    if geometry.is_file():
        mesh = GmshRunner(executable=gmsh_executable).run(
            geometry,
            output_dir=output_dir / "mesh",
        )
        text = (SIMCASES_DIR / f"{base_name}.i").read_text(encoding="utf-8")
        mesh_key = "mesh_file" if "mesh_file =" in text else "Mesh/file"
        parameters[mesh_key] = str(mesh.output)
    return MooseRunner(
        executable=moose_executable,
        mpi=mpi,
        n_tasks=n_tasks,
        n_threads=n_threads,
    ).run(deck, parameters=parameters, output_dir=output_dir / "solve")


def main() -> None:
    """Run one case without changing its packaged source files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", help="For example case18 or case00_HEX8")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "pyvale-output")
    parser.add_argument("--moose-executable", type=Path)
    parser.add_argument("--gmsh-executable", type=Path)
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    result = run_case(
        args.case,
        args.output_dir / args.case,
        moose_executable=args.moose_executable,
        gmsh_executable=args.gmsh_executable,
        mpi=args.mpi,
        n_tasks=args.tasks,
        n_threads=args.threads,
    )
    print(result.output)


if __name__ == "__main__":
    main()
