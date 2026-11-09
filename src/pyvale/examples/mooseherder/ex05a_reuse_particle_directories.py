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

# %%
# Case00_HEX8: 3D thermo-mechanical cube; m, Pa, seconds and degrees Celsius.
# This accessor points to pyvale/data/simulation/simcases/case00_HEX8.i.
input_path: Path = dataset.element_case_input_path(dataset.EElemTest.HEX8)
print(f"Packaged MOOSE input: {input_path}")

# %%
# Start a fresh session when rerunning this notebook. Within this session,
# repeated reuse IDs replace their previous evaluations.
example_path: Path = mh.get_next_run_path(output_path / "ex05a_particles")
manager: mh.DirectoryManager = mh.DirectoryManager(
    example_path,
    mode=mh.EDirMode.REUSE,
)
herd: mh.MooseHerd = mh.MooseHerd(
    [mh.SimStep("solve", mh.MooseRunner(), input_path)],
    dir_manager=manager,
    parallel_runners=2,
)
results: list[list[mh.RunResult]] = herd.run(
    [{"solve": {"EMod": 90e9}}, {"solve": {"EMod": 100e9}}],
    reuse_ids=["particle_0001", "particle_0007"],
)
for row in results:
    print(row[0].output_file)

# %%
# Submit only particle 7 next; particle 1's output remains intact.
results = herd.run([{"solve": {"EMod": 110e9}}], reuse_ids=["particle_0007"])
loader: mh.SweepLoader = mh.SweepLoader(
    manager.output_dir / "particle_0007/evaluation.json"
)
print("Latest particle 7 input_params:", loader.case_input_params)
print("Latest particle 7 output:", results[0][0].output_file)
