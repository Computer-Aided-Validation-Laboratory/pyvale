# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Chain mesh generation and Rabbit
================================

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
mesh_sources: tuple[Path, ...] = (geometry_path, input_path.with_name("case17_mesh.py"))
for source in mesh_sources:
    print(f"Mesh input: {source}")
    steps: list[mh.SimulationStep] = [
        mh.SimulationStep("mesh", mh.GmshRunner(), source),
        mh.SimulationStep("solve", mh.MooseRunner(), input_path, mesh_from="mesh"),
    ]
    mesh_parameters: dict[str, mh.MooseValue] = (
        {"hole_sect_nodes": 5, "plate_rad_nodes": 5}
        if source.suffix == ".geo"
        else {"mesh_size_mm": 20.0}
    )
    herd: mh.MooseHerd = mh.MooseHerd(
        steps,
        output_dir=output_path / f"ex07_{source.stem}",
    )
    results: list[list[mh.RunResult]] = herd.run(
        [
            {
                "mesh": mesh_parameters,
                "solve": {"EMod": 110e3, "endTime": 1, "timeStep": 1},
            },
        ]
    )
    print(results[0][-1].output)
