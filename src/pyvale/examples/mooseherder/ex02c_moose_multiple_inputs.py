# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""MOOSE: Ordered input decks
==========================

Run case00_HEX8 with its shared solver_smp.i supplied as a second entry deck.
MOOSE merges decks in order, so the final solver entry replaces the cube's
stricter nonlinear tolerances with standard defaults. CLI overrides take final
precedence.
"""

from pathlib import Path

import pyvale.data as dataset
import pyvale.mooseherder as mh

# %%
# Create the standard pyvale output directory for the simulation output.
output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

# %%
# Case00_HEX8: 3D thermo-mechanical cube; m, Pa, seconds and degrees Celsius.
# This accessor points to pyvale/data/simulation/simcases/case00_HEX8.i.
input_path: Path = dataset.element_case_input_path(dataset.EElemTest.HEX8)
print(f"Packaged MOOSE input: {input_path}")

# %%
# Generate the input file list and pass to the runner to run the simulation.
# This tuple is an ordered sequence of decks for ONE simulation: later decks
# override earlier ones. It is not a batch of independent simulation cases;
# that outer case list is introduced in the Sim Sweep examples.
input_files: tuple[Path, ...] = (input_path, dataset.sim_case_support_file_path("solver_smp.i"))
for entry in input_files:
    print(f"Entry deck: {entry}")

result: mh.RunResult = mh.MooseRunner().run(
    input_files,
    input_root=input_path.parent,
    input_params={"EMod": 110e9, "Executioner/nl_max_its": 25},
    output_dir=mh.get_next_run_path(output_path / "mooseherder_ex02c_solve"),
)
print(result.command)
