# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Mesh a Gmsh geometry file
=========================

Mesh the existing case17.geo plate with a hole in mm. Its named bc-base and
bc-top boundaries match case17.i's weak-plane-stress model. Gmsh DefineConstant
entries let CLI parameters change the mesh resolution; the geometry is unchanged.
The Python API runs in an isolated worker without opening a graphical window.
"""

from pathlib import Path

import pyvale.data as dataset
import pyvale.mooseherder as mh

# %%
output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

# %%
# Case17: 2D plate with a hole, weak plane stress; mm, MPa and seconds.
# These paths point to pyvale/data/simulation/simcases/case17.i and case17.geo.
input_path: Path = dataset.sim_case_input_file_path(17)
geo_file: Path | None = dataset.sim_case_gmsh_file_path(17)
assert geo_file is not None
print(f"Packaged MOOSE input: {input_path}")
print(f"Packaged Gmsh geometry: {geo_file}")

# %%
runner: mh.GmshRunner = mh.GmshRunner(geo_file=geo_file)
result: mh.RunResult = runner.run(
    input_params={"hole_sect_nodes": 7, "plate_rad_nodes": 7},
    output_dir=mh.get_next_run_path(output_path / "ex03a_mesh"),
)
print(f"Mesh: {result.output_file}")
