# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
MOOSE: CLI input parameters
===========================

Override case00_HEX8's named variables without modifying its source deck.
This 3D thermo-mechanical cube uses m, Pa, seconds and degrees Celsius. Native
MOOSE block paths also work. Here the thermal/mechanical history runs for five
seconds with one-second steps; endTime also sets the prescribed loading rate.
"""

from pathlib import Path

import pyvale.data as dataset
import pyvale.mooseherder as mh

# %%
# Create the standard pyvale output directory to send the simulation output to.
output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

# %%
# We will run a specific simulation from the pyvale simulation library:
# Case00_HEX8: 3D thermo-mechanical cube; m, Pa, seconds and degrees Celsius.
# This accessor points to pyvale/data/simulation/simcases/case00_HEX8.i.
input_path: Path = dataset.element_case_input_path(dataset.EElemTest.HEX8)
print(f"Packaged MOOSE input: {input_path}")

# %%
# This is the dictionary we will use to modify the moose input deck parameters through
# the cli. If we inspect the moose input file we will see these parameter names within
# the file. We can changes parameters within a [] block using the Block/parameter syntax
# as shown below:
input_params: dict[str, mh.InputValues] = {
    "EMod": 110e9,
    "nElemX": 3,
    "endTime": 5,
    "timeStep": 1,
    "Executioner/nl_max_its": 25,
}

# %%
# Run the simulation and inspect the command used to run it as well as the output log.
result: mh.RunResult = mh.MooseRunner(output_file_base="cube_parameters").run(
    input_path,
    input_params=input_params,
    output_dir=mh.get_next_run_path(output_path / "mooseherder_ex02a_solve"),
)

print(result.command)
print(result.log_file.read_text(encoding="utf-8")[-1000:])
