# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Sim Sweep: Read sweep results
=============================

Read completed case00_HEX8 cube outputs using a portable sweep.json manifest.
The example creates a small independent sweep first; opening its manifest does
not rerun anything. The packaged cube uses m, Pa, seconds and degrees Celsius.
"""

from pathlib import Path

import pyvale.data as dataset
import pyvale.dataio as io
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
# Create a small sweep so this example runs independently, then locate its
# completion manifest. SweepLoader reads the recorded output paths into
# SimData objects, allowing case parameters and field values to be compared.
herd: mh.MooseHerd = mh.MooseHerd(
    [mh.SimStep("solve", mh.MooseRunner(), input_path)],
    output_dir=output_path / "mooseherder_ex04d_sweep",
)

# %%
# The input list below requests TWO complete cases. Each case dictionary maps
# the stage name "solve" to a dictionary of parameter names and values.
# Results use results[case_index][stage_index], so results[0][0] is the first
# case's solver result. Its directory lets us locate the shared sweep manifest.
case_input_params: list[dict[str, dict[str, mh.InputValues]]] = [
    {"solve": {"EMod": 90e9}},
    {"solve": {"EMod": 110e9}},
]
results: list[list[mh.RunResult]] = herd.run(case_input_params)

# %%
# SweepLoader.case_input_params contains the original nested case dictionaries:
# case_input_params[case_index][stage_name][parameter_name]. load_sim_data("solve")
# selects ONE stage across all cases and returns a flat list of SimData objects,
# ordered by case. That is why zip() pairs these two collections correctly.
manifest: Path = results[0][0].output_dir.parents[1] / "sweep.json"

loader: mh.SweepLoader = mh.SweepLoader(manifest)
simulations: list[io.SimData] = loader.load_sim_data("solve")

for sim_case, sim_data in zip(loader.case_input_params, simulations):
    print(sim_case, sim_data.time, sim_data.node_vars["temperature"][:, -1].max())
