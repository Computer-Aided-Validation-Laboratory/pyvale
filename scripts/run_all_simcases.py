# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Run every numbered packaged simulation case using the default Rabbit build."""

from pathlib import Path

from run_simcase import run_case


def main() -> None:
    """Keep each case's mesh, log and solve beneath pyvale-output."""
    for number in range(1, 27):
        name = f"case{number:02d}"
        print(run_case(name, Path.cwd() / "pyvale-output" / name).output_file)


if __name__ == "__main__":
    main()
