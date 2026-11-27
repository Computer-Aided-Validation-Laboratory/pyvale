# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Sim Sweep: Parallel parameter sweep
===================================

Run independent case00_HEX8 thermo-mechanical cubes concurrently, each with
one process and one thread. This requires no MPI. Cube modulus values are Pa;
the rest of the packaged loading and 20-second history are retained.
"""

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
# Case00_HEX8: 3D thermo-mechanical cube; m, Pa, seconds and degrees Celsius.
# This accessor points to pyvale/data/simulation/simcases/case00_HEX8.i.
input_path: Path = dataset.element_case_input_path(dataset.EElemTest.HEX8)
print(f"Packaged MOOSE input: {input_path}")

# %%
# Evaluate three elastic moduli using at most two concurrent runner chains.
# Each case owns a separate output directory, and returned results retain
# the submitted case order even when simulations finish out of order.
herd: mh.MooseHerd = mh.MooseHerd(
    [mh.SimStep("solve", mh.MooseRunner(), input_path)],
    output_dir=output_path / "mooseherder_ex04c_sweep",
    parallel_runners=2,
)

# %%
# The outer list contains THREE independent cases, one per modulus. Each case
# is a dictionary keyed by the stage name "solve"; its inner dictionary maps
# parameter names to values. Thus case_input_params[1]["solve"]["EMod"] is 100e9.
# The list is what makes this a batch, rather than one run with several stages.
case_input_params: list[dict[str, dict[str, mh.InputValues]]] = [
    {"solve": {"EMod": modulus}} for modulus in (90e9, 100e9, 110e9)
]

# %%
# Returned results are indexed first by case, then by stage. Each case here
# has only one stage, so results[i][0] is its solver RunResult. The outer list
# retains input order despite concurrent execution, allowing zip() to pair each
# case dictionary with the correct result row.
results: list[list[mh.RunResult]] = herd.run(case_input_params)

for case, row in zip(case_input_params, results):
    print(case, row[-1].output_file)
