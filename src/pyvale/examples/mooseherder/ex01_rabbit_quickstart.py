# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Rabbit FEM quickstart
=====================

Run a small plane-strain tensile specimen with the installed serial/SMP Rabbit
package. The model uses SI units: metres, seconds and pascals. No MOOSE source
checkout or executable configuration is needed.
"""

from pathlib import Path

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh

# %%
# Every example is independent and writes below the standard output directory.
output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)
data_path: Path = dataset.mooseherder_data_dir()

# %%
# Run the default input deck and inspect its nodal displacements.
runner: mh.MooseRunner = mh.MooseRunner()
result: mh.RunResult = runner.run(
    data_path / "model.i",
    output_dir=mh.next_run_path(output_path / "ex01_solve"),
)
reader: mh.ExodusLoader = mh.ExodusLoader(result.output)
sim_data: io.SimData = reader.load_all_sim_data()
reader.close()
print(f"Output: {result.output}")
print(f"Times: {sim_data.time}")
print(f"Nodal fields: {tuple(sim_data.node_vars)}")
