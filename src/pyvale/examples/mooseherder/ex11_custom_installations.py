# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Select custom MOOSE and Gmsh installations
==========================================

Explicit executable paths take precedence over packaged defaults. A custom app
may supply additional environment variables without changing the parent Python
process. Leave the paths as None to run this example using installed defaults.
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
# Replace None with Path("/absolute/path/to/custom-opt") or your Gmsh binary.
moose_executable: Path | None = None
gmsh_executable: Path | None = None
mesh_runner: mh.GmshRunner = mh.GmshRunner(executable=gmsh_executable)
mesh: mh.RunResult = mesh_runner.run(
    data_path / "specimen.geo",
    output_dir=mh.next_run_path(output_path / "ex11_mesh"),
)
runner: mh.MooseRunner = mh.MooseRunner(executable=moose_executable, env={})
result: mh.RunResult = runner.run(
    data_path / "mesh_model.i",
    parameters={"Mesh/file": str(mesh.output)},
    output_dir=mh.next_run_path(output_path / "ex11_solve"),
)
print(result.output)
