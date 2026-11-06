# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Nested MOOSE includes
=====================

The entry deck includes parameters.i and physics.i; physics.i includes
materials/elastic.i. Staging preserves the whole input tree. MOOSE resolves
includes and CLI overrides; PyVale does not rewrite or flatten the deck.
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
# EMod is defined in parameters.i and referenced in materials/elastic.i.
result: mh.RunResult = mh.MooseRunner().run(
    data_path / "model.i",
    input_root=data_path,
    parameters={"EMod": 2e9},
    output_dir=mh.next_run_path(output_path / "ex03_solve"),
)
assert (result.directory / "inputs/materials/elastic.i").is_file()
print(result.directory / "parameters.json")
