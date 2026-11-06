# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Mesh with a supplied Python script
==================================

A mesh script receives --config followed by a JSON file. It reads parameters
and output_path, uses the Gmsh Python API, and writes the declared mesh. Inspect
the staged mesh_specimen.py to see a complete implementation of this contract.
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
runner: mh.GmshRunner = mh.GmshRunner(script=data_path / "mesh_specimen.py")
result: mh.RunResult = runner.run(
    parameters={"n": 5},
    output_dir=mh.next_run_path(output_path / "ex06_mesh"),
)
print((result.directory / "mesh-config.json").read_text(encoding="utf-8"))
print(result.output)
