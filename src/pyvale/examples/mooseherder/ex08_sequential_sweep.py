# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Sequential parameter sweep
==========================

Sweep mesh density and material stiffness through a mesh/solve chain. Case
parameters are keyed by step name. Every case has a separate working directory;
source geometry, scripts and decks remain unchanged.
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
steps: list[mh.SimulationStep] = [
    mh.SimulationStep("mesh", mh.GmshRunner(), data_path / "specimen.geo"),
    mh.SimulationStep(
        "solve",
        mh.MooseRunner(),
        data_path / "mesh_model.i",
        mesh_from="mesh",
    ),
]
cases: list[dict[str, dict[str, mh.MooseValue]]] = mh.sweep_param_grid(
    {
        "mesh": {"n": [3, 4]},
        "solve": {"EMod": [1e9, 2e9]},
    }
)
herd: mh.MooseHerd = mh.MooseHerd(steps, output_dir=output_path / "ex08_sweep")
results: list[list[mh.RunResult]] = herd.run(cases)
print(f"Completed {len(results)} cases")
