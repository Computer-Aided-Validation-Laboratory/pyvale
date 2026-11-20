# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Gmsh: Custom installation
=========================

Mesh the packaged case17 plate with a hole using an explicit Gmsh executable.
Leave the executable as None to use the installed Python Gmsh package. The
geometry uses mm. An invalid custom executable never falls back to Python Gmsh.
Custom Gmsh executables support .geo files; Python mesh scripts run with Python.
"""

from pathlib import Path

import pyvale.data as dataset
import pyvale.mooseherder as mh

# %%
# Create the standard output directory so generated files stay together
# under pyvale-output rather than alongside the packaged input files.
# Locate the packaged plate geometry to test the selected Gmsh executable.
output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

geo_file: Path | None = dataset.sim_case_gmsh_file_path(17)
assert geo_file is not None
print(f"Packaged Gmsh geometry: {geo_file}")

# %%
# Set this to your Gmsh executable, for example Path('/opt/gmsh/bin/gmsh').
gmsh_exe_file: Path | None = None

# %%
# Export an Abaqus .inp mesh: the extension selects the format. Use .msh with
# msh_version=2.2 when passing the mesh to the bundled MOOSE models.
result: mh.RunResult = mh.GmshRunner(
    gmsh_exe_file=gmsh_exe_file, output_file_name="plate.inp"
).run(
    geo_file,
    output_dir=mh.get_next_run_path(output_path / "mooseherder_ex03c_mesh"),
)
print(result.output_file)
