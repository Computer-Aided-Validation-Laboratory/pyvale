# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Explicit textual input modification
===================================

CLI parameters are the default for sweeps. InputModifier is still useful when
you deliberately want an edited input file. It edits only a marked variable
block in one file; it does not recursively parse includes or modify scripts.
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
# Copy the input tree so the edited parameter file keeps its include context.
import shutil

input_tree: Path = output_path / "ex13_inputs"
shutil.copytree(data_path, input_tree, dirs_exist_ok=True)
modifier: mh.InputModifier = mh.InputModifier(input_tree / "parameters.i", "#", "")
modifier.update_vars({"EMod": 2e9})
modifier.write_file(input_tree / "parameters.i")
result: mh.RunResult = mh.MooseRunner().run(
    input_tree / "model.i",
    output_dir=mh.next_run_path(output_path / "ex13_solve"),
)
print(result.output)
