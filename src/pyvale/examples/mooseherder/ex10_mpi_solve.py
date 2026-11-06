# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Opt into MPI
============

Select the existing case00_HEX8.i or case00_HEX8_mpi.i cube with the dataset
accessor's mpi flag. The default remains serial/SMP. Set USE_MPI only with an
MPI-capable Rabbit installation and a compatible launcher; the MPI solver
uses Hypre. Both entries share the same physics, units and loading history.
"""

from pathlib import Path

import pyvale.data as dataset
import pyvale.mooseherder as mh

# %%
output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

# %%
USE_MPI = False
input_path: Path = dataset.element_case_input_path(dataset.EElemTest.HEX8, mpi=USE_MPI)
print(f"Packaged MOOSE input: {input_path}")

# %%
runner: mh.MooseRunner = mh.MooseRunner(mpi=USE_MPI, n_tasks=2 if USE_MPI else 1)
result: mh.RunResult = runner.run(
    input_path,
    output_dir=mh.next_run_path(output_path / "ex10_solve"),
)
print(result.output)
