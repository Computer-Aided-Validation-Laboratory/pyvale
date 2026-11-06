# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Reuse directories for optimiser particles
=========================================

Use the existing case00_HEX8 thermo-mechanical cube to demonstrate stable
particle identities. Modulus is in Pa. REUSE replaces only the selected
particle's previous evaluation; ARCHIVE remains the default for normal sweeps.
Read each result before submitting the same ID again.
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
manager: mh.DirectoryManager = mh.DirectoryManager(
    output_path / "ex14_particles",
    mode=mh.EDirectoryMode.REUSE,
)
herd: mh.MooseHerd = mh.MooseHerd(
    [mh.SimulationStep("solve", mh.MooseRunner(), input_path)],
    directory_manager=manager,
    num_parallel=2,
)
results: list[list[mh.RunResult]] = herd.run(
    [{"solve": {"EMod": 90e9}}, {"solve": {"EMod": 100e9}}],
    evaluation_ids=["particle-0001", "particle-0007"],
)
for row in results:
    print(row[0].output)

# %%
# Submit only particle 7 next; particle 1's output remains intact.
results = herd.run([{"solve": {"EMod": 110e9}}], evaluation_ids=["particle-0007"])
loader: mh.SweepLoader = mh.SweepLoader(
    manager.output_dir / "particle-0007/evaluation.json"
)
print("Latest particle 7 parameters:", loader.cases)
print("Latest particle 7 output:", results[0][0].output)
