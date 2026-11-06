# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Rabbit FEM quickstart
=====================

Run PyVale's existing case00_HEX8 thermo-mechanical cube with installed Rabbit.
The cube uses metres, pascals, seconds and degrees Celsius, and generates its
mesh internally. The example prints the packaged .i path so you can inspect
its loads, materials, fields and included solver settings.
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
# Read all fields into a typed SimData container after the solve completes.
import pyvale.dataio as io

result: mh.RunResult = mh.MooseRunner().run(
    input_path,
    output_dir=mh.next_run_path(output_path / "ex01_solve"),
)
reader: mh.ExodusLoader = mh.ExodusLoader(result.output)
try:
    sim_data: io.SimData = reader.load_all_sim_data()
finally:
    reader.close()
print(f"Output: {result.output}")
print(f"Times: {sim_data.time}")
print(f"Nodal fields: {tuple(sim_data.node_vars)}")
