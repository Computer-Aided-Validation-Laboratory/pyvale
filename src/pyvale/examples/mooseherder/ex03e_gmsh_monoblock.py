# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Gmsh: Mesh the case16 three-material monoblock
==============================================

Generate the authoritative case16 3D divertor monoblock mesh without changing
its geometry, material regions or boundary names. The model has tungsten
armour, a copper interlayer and a CuCrZr coolant tube; the coolant bore is a
void rather than a fourth material. Geometry is in metres.

The existing geometry extrudes a mapped cross-section into quadratic hexahedra.
It owns its mesh resolution and element-order settings. This example uses
those defaults and only generates the mesh, not the thermo-mechanical solution.
"""

from pathlib import Path

import pyvale.data as dataset
import pyvale.mooseherder as mh

# %%
# Resolve the registered geometry and matching input deck through PyVale's data
# API. Files are downloaded and cached when not already available locally.
geo_file: Path | None = dataset.sim_case_gmsh_file_path(16)
if geo_file is None:
    raise FileNotFoundError("The case16 Gmsh geometry is unavailable")
input_file: Path = dataset.sim_case_input_file_path(16)
print(f"Case16 geometry: {geo_file}")
print(f"Matching thermo-mechanical input: {input_file}")

# %%
# GmshRunner stages the geometry before executing its existing Mesh 3 command.
# The source and its mesh controls are preserved. Material physical groups are
# armour-w, interlayer-cu and pipe-cucrzr; case16.i uses these exact names.
runner = mh.GmshRunner(geo_file=geo_file, mesh_dims=3)
result: mh.RunResult = runner.run(
    output_dir=mh.get_next_run_path(
        Path.cwd() / "pyvale-output" / "mooseherder_ex03e_mesh"
    ),
)
print(f"Three-material monoblock mesh: {result.output_file}")
