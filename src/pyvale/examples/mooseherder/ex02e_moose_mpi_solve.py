# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
MOOSE: MPI execution
====================

Select the existing case00_HEX8.i or case00_HEX8_mpi.i cube with the dataset
accessor's mpi flag. The default remains serial/SMP. Set N_TASKS above 1 with an
MPI-capable Rabbit installation and a compatible launcher; the MPI solver
uses Hypre. Both entries share the same physics, units and loading history.
"""

import sys
from pathlib import Path

import pyvale.data as dataset
import pyvale.mooseherder as mh

# %%
# Create the standard output directory so generated files stay together
# under pyvale-output rather than alongside the packaged input files.
output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

# %%
# Choose the solver deck to match the requested MPI rank count. None or
# 1 selects the serial/SMP deck; larger counts require an MPI-capable
# Rabbit installation and a matching MPI launcher.
N_TASKS: int | None = None
input_path: Path = dataset.element_case_input_path(
    dataset.EElemTest.HEX8, mpi=N_TASKS is not None and N_TASKS > 1
)
print(f"Packaged MOOSE input: {input_path}")

# %%
# Select the launcher installed for the MPI runtime used to build the solver.
# Windows commonly uses mpiexec; on Linux/macOS choose Open MPI or set USE_MPICH
# for MPICH. MPICH is the MPI implementation, and its launcher is mpiexec.
# The runner appends N_TASKS after the rank-count option, followed by the solver.
# These launcher settings are unused when N_TASKS is None or 1.
USE_MPICH: bool = False
mpi_launcher: tuple[str, ...]
if sys.platform == "win32" or USE_MPICH:
    mpi_launcher = ("mpiexec", "-n")
else:
    mpi_launcher = ("mpirun", "-np")

# %%
# Pass the same rank count to the runner so execution matches the selected
# deck. Keep earlier results by selecting a new directory for each run.
runner: mh.MooseRunner = mh.MooseRunner(n_tasks=N_TASKS, launcher=mpi_launcher)
result: mh.RunResult = runner.run(
    input_path,
    output_dir=mh.get_next_run_path(output_path / "mooseherder_ex02e_solve"),
)
print(result.output_file)
