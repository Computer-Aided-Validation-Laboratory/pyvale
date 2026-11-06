# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Mesh with a supplied Python script
==================================

Use case17_mesh.py, packaged beside case17.i, as a Python Gmsh alternative to
case17.geo. It builds the same 100 x 150 mm plate and 12.5 mm-radius central
hole with the same boundary names. The .geo mesh uses mapped quadrilaterals;
the Python script uses triangles, both with quadratic geometry by default.

The script receives --config followed by a JSON file containing parameters and
output_path. Open the printed script path to inspect the complete implementation.
"""

from pathlib import Path

import pyvale.data as dataset
import pyvale.mooseherder as mh

# %%
output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

# Case17: 2D plate with a hole, weak plane stress; mm, MPa and seconds.
# These paths point to pyvale/data/simulation/simcases/case17.i and case17.geo.
input_path: Path = dataset.sim_case_input_file_path(17)
geometry_path: Path | None = dataset.sim_case_gmsh_file_path(17)
assert geometry_path is not None
print(f"Packaged MOOSE input: {input_path}")
print(f"Packaged Gmsh geometry: {geometry_path}")

# %%
script_path: Path = input_path.with_name("case17_mesh.py")
print(f"Packaged Python mesh script: {script_path}")
runner: mh.GmshRunner = mh.GmshRunner(script=script_path)
result: mh.RunResult = runner.run(
    parameters={"mesh_size_mm": 15.0},
    output_dir=mh.next_run_path(output_path / "ex06_mesh"),
)
print((result.directory / "mesh-config.json").read_text(encoding="utf-8"))
print(result.output)
