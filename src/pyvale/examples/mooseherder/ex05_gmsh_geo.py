# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Mesh a Gmsh geometry file
=========================

Generate a transfinite quadrilateral mesh from a .geo file with named physical
boundaries. Gmsh DefineConstant parameters accept CLI overrides. The Python API
runs in an isolated worker and never opens a graphical window.
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
runner: mh.GmshRunner = mh.GmshRunner(geometry=data_path / "specimen.geo")
result: mh.RunResult = runner.run(
    parameters={"n": 5},
    output_dir=mh.next_run_path(output_path / "ex05_mesh"),
)
print(f"Mesh: {result.output}")
