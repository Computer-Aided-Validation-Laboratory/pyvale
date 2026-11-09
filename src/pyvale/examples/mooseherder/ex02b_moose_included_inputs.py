# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""MOOSE: Included input decks
===========================

Use case00_HEX8's existing include hierarchy: case00_HEX8.i includes
solver_cube_smp.i, which includes solver_smp.i. The run snapshots the packaged
simcases tree and preserves relative paths. MOOSE resolves the includes and
CLI overrides; PyVale does not rewrite or flatten the deck.
"""

from pathlib import Path

import pyvale.data as dataset
import pyvale.mooseherder as mh

# %%
# Create the standard pyvale output directory to send the simulation output to.
output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

# %%
# Case00_HEX8: 3D thermo-mechanical cube; m, Pa, seconds and degrees Celsius.
# This accessor points to pyvale/data/simulation/simcases/case00_HEX8.i.
input_path: Path = dataset.element_case_input_path(dataset.EElemTest.HEX8)
print(f"Packaged MOOSE input: {input_path}")

# %%
# Inpsect the input decks and the includes for the input deck before passing to the
# runner to run the simulation.
input_root: Path = input_path.parent
print(f"Cube solver include: {input_root / 'solver_cube_smp.i'}")
print(f"Base solver include: {input_root / 'solver_smp.i'}")

result: mh.RunResult = mh.MooseRunner().run(
    input_path,
    input_root=input_root,
    input_params={"EMod": 110e9},
    output_dir=mh.get_next_run_path(output_path / "mooseherder_ex02b_solve"),
)

assert (result.output_dir / "input_files/solver_cube_smp.i").is_file()
assert (result.output_dir / "input_files/solver_smp.i").is_file()
print(result.output_dir / "input_params.json")
