# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Sequential parameter sweep
==========================

Sweep the existing case17 plate's mapped mesh resolution and elastic modulus
through a mesh/solve chain. Geometry is in mm and modulus in MPa. Each case
has its own working directory; the packaged case17.geo and case17.i stay
unchanged. One time step reaches the prescribed 1 mm final displacement.
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
steps: list[mh.SimulationStep] = [
    mh.SimulationStep("mesh", mh.GmshRunner(), geometry_path),
    mh.SimulationStep("solve", mh.MooseRunner(), input_path, mesh_from="mesh"),
]
cases: list[dict[str, dict[str, mh.MooseValue]]] = mh.sweep_param_grid(
    {
        "mesh": {"hole_sect_nodes": [5, 7], "plate_rad_nodes": [5]},
        "solve": {"EMod": [90e3, 110e3], "endTime": [1], "timeStep": [1]},
    }
)
herd: mh.MooseHerd = mh.MooseHerd(steps, output_dir=output_path / "ex08_sweep")
results: list[list[mh.RunResult]] = herd.run(cases)
print(f"Completed {len(results)} cases")
