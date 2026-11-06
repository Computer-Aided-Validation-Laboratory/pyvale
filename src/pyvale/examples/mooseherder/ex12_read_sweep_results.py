# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Read completed sweep results
============================

SweepLoader reads a portable sweep.json manifest and loads a named solver
step's Exodus files in original case order. This example first creates a small
sweep so it runs independently; reopening the manifest does not rerun anything.
"""

from pathlib import Path

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh

# %%
# Every example is independent and writes below the standard output directory.
output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)
data_path: Path = dataset.mooseherder_data_dir()

# %%
herd: mh.MooseHerd = mh.MooseHerd(
    [mh.SimulationStep("solve", mh.MooseRunner(), data_path / "model.i")],
    output_dir=output_path / "ex12_sweep",
)
results: list[list[mh.RunResult]] = herd.run(
    [
        {"solve": {"topDisp": 1e-4}},
        {"solve": {"topDisp": 2e-4}},
    ]
)
manifest: Path = results[0][0].directory.parents[1] / "sweep.json"
loader: mh.SweepLoader = mh.SweepLoader(manifest)
simulations: list[io.SimData] = loader.load_sim_data("solve")
for case, simulation in zip(loader.cases, simulations):
    print(case, simulation.node_vars["disp_y"][:, -1].max())
