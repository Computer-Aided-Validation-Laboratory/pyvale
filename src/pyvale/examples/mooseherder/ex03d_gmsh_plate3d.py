# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Gmsh: Mesh the packaged 3D plate with a hole
==========================================

Use the existing case08 geometry from PyVale's registered data library. It
extrudes a 100 x 150 mm plate with a 12.5 mm-radius hole through 5 mm thickness.
Native coordinates are metres. Preserve its quadratic hexahedra and named
plate, bc-base and bc-top physical groups. No solver is launched.
"""

from pathlib import Path

import pyvale.data as dataset
import pyvale.mooseherder as mh

# %%
# The data API fetches and caches the authoritative geometry through Pooch.
geo_file: Path | None = dataset.sim_case_gmsh_file_path(8)
if geo_file is None:
    raise FileNotFoundError("The case08 Gmsh geometry is unavailable")
print(f"Packaged 3D geometry: {geo_file}")

# %%
# Use the geometry's existing Mesh 3 and element-order controls unchanged.
# GmshRunner executes a staged copy and exports MSH 2.2 by default.
runner = mh.GmshRunner(geo_file=geo_file, mesh_dims=3)
result: mh.RunResult = runner.run(
    output_dir=mh.get_next_run_path(
        Path.cwd() / "pyvale-output" / "mooseherder_ex03d_mesh"
    ),
)
print(f"3D plate mesh: {result.output_file}")
