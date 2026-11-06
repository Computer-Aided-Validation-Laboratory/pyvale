# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Select custom MOOSE and Gmsh installations
==========================================

Select explicit MOOSE and Gmsh executables for the existing case17 plate
with a hole. Leave the paths as None to use the installed defaults. The deck
uses mm, MPa, weak plane stress and a prescribed 1 mm displacement. The example
uses one time step; custom executable paths never fall back silently.
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
moose_executable: Path | None = None
gmsh_executable: Path | None = None
mesh: mh.RunResult = mh.GmshRunner(executable=gmsh_executable).run(
    geometry_path,
    output_dir=mh.next_run_path(output_path / "ex11_mesh"),
)
result: mh.RunResult = mh.MooseRunner(executable=moose_executable, env={}).run(
    input_path,
    parameters={"Mesh/file": str(mesh.output), "endTime": 1, "timeStep": 1},
    output_dir=mh.next_run_path(output_path / "ex11_solve"),
)
print(result.output)
