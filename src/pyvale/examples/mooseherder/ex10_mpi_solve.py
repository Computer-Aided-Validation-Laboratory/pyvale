# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Opt into MPI
============

The default remains serial/SMP. Set USE_MPI to True only after installing an
MPI-capable Rabbit build and a compatible launcher. Select the matching solver
entry deck as well as the launch policy. Multiple MPI ranks cannot be requested
from a serial Rabbit installation.
"""

from pathlib import Path

import pyvale.data as dataset
import pyvale.mooseherder as mh

# %%
# Every example is independent and writes below the standard output directory.
output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)
data_path: Path = dataset.mooseherder_data_dir()

# %%
# Leave False for unattended gallery execution and ordinary installations.
USE_MPI = False
runner: mh.MooseRunner = mh.MooseRunner(
    mpi=USE_MPI,
    n_tasks=2 if USE_MPI else 1,
)
input_path: Path = data_path / ("model_mpi.i" if USE_MPI else "model.i")
result: mh.RunResult = runner.run(
    input_path,
    output_dir=mh.next_run_path(output_path / "ex10_solve"),
)
print(result.output)
