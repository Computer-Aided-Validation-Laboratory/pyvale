# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Read completed sweep results
============================

Read completed case00_HEX8 cube outputs using a portable sweep.json manifest.
The example creates a small independent sweep first; opening its manifest does
not rerun anything. The packaged cube uses m, Pa, seconds and degrees Celsius.
"""

from pathlib import Path

import pyvale.data as dataset
import pyvale.mooseherder as mh

# %%
output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

# Case00_HEX8: 3D thermo-mechanical cube; m, Pa, seconds and degrees Celsius.
# This accessor points to pyvale/data/simulation/simcases/case00_HEX8.i.
input_path: Path = dataset.element_case_input_path(dataset.EElemTest.HEX8)
print(f"Packaged MOOSE input: {input_path}")

# %%
import pyvale.dataio as io

herd: mh.MooseHerd = mh.MooseHerd(
    [mh.SimulationStep("solve", mh.MooseRunner(), input_path)],
    output_dir=output_path / "ex12_sweep",
)
results: list[list[mh.RunResult]] = herd.run(
    [
        {"solve": {"EMod": 90e9}},
        {"solve": {"EMod": 110e9}},
    ]
)
manifest: Path = results[0][0].directory.parents[1] / "sweep.json"
loader: mh.SweepLoader = mh.SweepLoader(manifest)
simulations: list[io.SimData] = loader.load_sim_data("solve")
for case, simulation in zip(loader.cases, simulations):
    print(case, simulation.time, simulation.node_vars["temperature"][:, -1].max())
