# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Chain mesh generation and Rabbit
================================

Run the same tensile model with a geometry file and a Python mesh script.
Each mesh is generated first; mesh_from passes its actual output path to the
MOOSE FileMesh parameter. A failed mesh prevents the solve from starting.
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
for source in ("specimen.geo", "mesh_specimen.py"):
    steps: list[mh.SimulationStep] = [
        mh.SimulationStep("mesh", mh.GmshRunner(), data_path / source),
        mh.SimulationStep(
            "solve",
            mh.MooseRunner(),
            data_path / "mesh_model.i",
            mesh_from="mesh",
        ),
    ]
    herd: mh.MooseHerd = mh.MooseHerd(
        steps,
        output_dir=output_path / f"ex07_{Path(source).stem}",
    )
    results: list[list[mh.RunResult]] = herd.run(
        [
            {"mesh": {"n": 4}, "solve": {"EMod": 2e9}},
        ]
    )
    print(results[0][-1].output)
