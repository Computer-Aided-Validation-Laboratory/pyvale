# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Sim Sweep: Chain meshing and solving
====================================

Solve the existing case17 weak-plane-stress plate using both its .geo mesh
and the alternative case17_mesh.py script. Coordinates are mm and modulus is
MPa. Each chain meshes first, then passes the resulting path to Mesh/file.
A failed mesh stops that chain before the solve begins. One time step reaches
the original final displacement of 1 mm for this elastic demonstration.
"""

from pathlib import Path

import pyvale.data as dataset
import pyvale.mooseherder as mh

# %%
# Create the standard output directory so generated files stay together
# under pyvale-output rather than alongside the packaged input files.
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
# Build the same mesh-to-solver chain for each mesh source. The solve step
# receives that case's generated mesh through mesh_from, so both the .geo
# and Python meshing workflows can be compared independently.
# The steps list gives execution order; each step's name_subdir ("mesh" or
# "solve") is also the key used in that case's parameter dictionary.
#
# herd.run() takes a list of cases. Here the list has ONE entry, containing a
# stage dictionary: "mesh" maps to mesh parameters and "solve" to solver
# parameters. Each stage's inner dictionary maps parameter names to values.
# Even a single case needs the outer list; the two mesh sources are evaluated
# by separate herd.run() calls in this loop, rather than being two stages.
#
# The returned list is indexed as results[case_index][stage_index]. For this
# one-case, two-stage chain, results[0][0] is the mesh RunResult and results[0][1]
# (or results[0][-1]) is the solve RunResult.
mesh_sources: tuple[Path, ...] = (geo_file, dataset.sim_case_support_file_path("case17_mesh.py"))
for source in mesh_sources:
    print(f"Mesh input: {source}")
    steps: list[mh.SimStep] = [
        mh.SimStep("mesh", mh.GmshRunner(), source),
        mh.SimStep("solve", mh.MooseRunner(), input_path, mesh_from="mesh"),
    ]

    mesh_input_params: dict[str, mh.InputValues] = (
        {"hole_sect_nodes": 5, "plate_rad_nodes": 5}
        if source.suffix == ".geo"
        else {"mesh_size_mm": 20.0}
    )

    herd: mh.MooseHerd = mh.MooseHerd(
        steps,
        output_dir=output_path / f"mooseherder_ex04a_{source.stem}",
    )

    case_input_params: list[dict[str, dict[str, mh.InputValues]]] = [
        {
            "mesh": mesh_input_params,
            "solve": {"EMod": 110e3, "endTime": 1, "timeStep": 1},
        },
    ]
    results: list[list[mh.RunResult]] = herd.run(case_input_params)

    print(results[0][-1].output_file)
