# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""
Quick start: Rabbit FEM
=======================

Run PyVale's existing case00_HEX8 thermo-mechanical cube with installed Rabbit.
The cube uses metres, pascals, seconds and degrees Celsius, and generates its
mesh internally. The example prints the packaged .i path so you can inspect
its loads, materials, fields and included solver settings.
"""

from pathlib import Path

import pyvale.data as dataset
import pyvale.dataio as io
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
print()

# %%
# Run the cube simulation and inspect the returned RunResult dataclass.

result: mh.RunResult = mh.MooseRunner().run(
    input_path,
    output_dir=mh.get_next_run_path(output_path / "mooseherder_ex01a_solve"),
)

print("RunResult:")
print(f"  output_file: {result.output_file}")
print(f"  output_dir: {result.output_dir}")
print(f"  command: {result.command}")
print(f"  log_file: {result.log_file}")
print()

# %%
# Load the output simulation data and inspect it.
with mh.ExodusLoader(result.output_file) as loader:
    sim_data: io.SimData = loader.load_all_sim_data()


print("SimData:")
print(f"Times: {sim_data.time}")
print(f"Nodal fields: {tuple(sim_data.node_vars)}")
print()
