# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Parallel independent SMP solves
===============================

Run independent Rabbit simulations concurrently, each with one thread and one
process. This does not require MPI. The orchestrator uses threads to launch
isolated subprocesses, so notebook users do not need a multiprocessing guard.
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
steps: list[mh.SimulationStep] = [
    mh.SimulationStep("solve", mh.MooseRunner(), data_path / "model.i"),
]
herd: mh.MooseHerd = mh.MooseHerd(
    steps,
    output_dir=output_path / "ex09_sweep",
    num_parallel=2,
)
cases: list[dict[str, dict[str, mh.MooseValue]]] = [
    {"solve": {"EMod": modulus}} for modulus in (1e9, 2e9, 3e9)
]
results: list[list[mh.RunResult]] = herd.run(cases)
for case, row in zip(cases, results):
    print(case, row[-1].output)
