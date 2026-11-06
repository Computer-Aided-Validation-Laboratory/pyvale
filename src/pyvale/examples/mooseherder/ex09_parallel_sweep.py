# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Parallel independent SMP solves
===============================

Run independent case00_HEX8 thermo-mechanical cubes concurrently, each with
one process and one thread. This requires no MPI. Cube modulus values are Pa;
the rest of the packaged loading and 20-second history are retained.
"""

from pathlib import Path

import pyvale.data as dataset
import pyvale.mooseherder as mh

# %%
output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

# Case00_HEX8: 3D thermo-mechanical cube; m, Pa, seconds and degrees Celsius.
# This accessor points to pyvale/data/simulation/simcases/case00_HEX8.i.
input_path: Path = dataset.element_case_input_path(dataset.EElemTest.HEX8)
print(f"Packaged MOOSE input: {input_path}")

# %%
herd: mh.MooseHerd = mh.MooseHerd(
    [mh.SimulationStep("solve", mh.MooseRunner(), input_path)],
    output_dir=output_path / "ex09_sweep",
    num_parallel=2,
)
cases: list[dict[str, dict[str, mh.MooseValue]]] = [
    {"solve": {"EMod": modulus}} for modulus in (90e9, 100e9, 110e9)
]
results: list[list[mh.RunResult]] = herd.run(cases)
for case, row in zip(cases, results):
    print(case, row[-1].output)
