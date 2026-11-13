# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Gmsh: Build a simple 3D mesh
=============================

Mesh a 20 x 10 x 5 mm box with linear tetrahedra using GmshRunner. The geometry
defines a solid material region and the fixed/loaded end faces, ready for a
solver to assign materials and boundary conditions. No solver is launched.

The companion geometry is available with this example's installed source.
Copy it beside this script if running a separately downloaded script.
"""

from pathlib import Path

import pyvale.mooseherder as mh

# %%
# Inspect the geometry, physical groups and overridable mesh size.
geo_file: Path = Path(__file__).with_name("gmsh_box.geo")
print(geo_file.read_text(encoding="utf-8"))

# %%
# Run headlessly in an isolated output directory. GmshRunner uses the installed
# Python Gmsh package and exports MSH 2.2, the default for MOOSE compatibility.
runner = mh.GmshRunner(geo_file=geo_file, mesh_dims=3, mesh_order=1)
result: mh.RunResult = runner.run(
    input_params={"mesh_size_mm": 2.5},
    output_dir=mh.get_next_run_path(
        Path.cwd() / "pyvale-output" / "mooseherder_ex03d_mesh"
    ),
)
print(f"3D box mesh: {result.output_file}")
