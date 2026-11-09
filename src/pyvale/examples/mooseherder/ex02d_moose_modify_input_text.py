# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""MOOSE: Modify input text
========================

Edit a copy of the existing case00_HEX8.i cube's marked variable block.
EMod is in Pa. CLI parameters remain the default for sweeps; InputModifier
is for deliberate text edits and does not recursively parse included files.
Copying the packaged simcases tree preserves its solver include paths.
"""

import shutil
from pathlib import Path

import pyvale.data as dataset
import pyvale.mooseherder as mh

# %%
# Create the standard output directory so generated files stay together
# under pyvale-output rather than alongside the packaged input files.
output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

# %%
# Case00_HEX8: 3D thermo-mechanical cube; m, Pa, seconds and degrees Celsius.
# This accessor points to pyvale/data/simulation/simcases/case00_HEX8.i.
input_path: Path = dataset.element_case_input_path(dataset.EElemTest.HEX8)
print(f"Packaged MOOSE input: {input_path}")

# %%
# Copy the complete input tree before editing so the packaged model stays
# unchanged and relative includes still resolve. Modify only the copied
# material parameter, then solve it in a separate output directory.

input_tree: Path = mh.get_next_run_path(output_path / "mooseherder_ex02d_inputs")
shutil.copytree(input_path.parent, input_tree)

edited_input: Path = input_tree / input_path.name
modifier: mh.InputModifier = mh.InputModifier(edited_input, "#", "")
modifier.update_vars({"EMod": 110e9})
modifier.write_file(edited_input)

result: mh.RunResult = mh.MooseRunner().run(
    edited_input,
    output_dir=mh.get_next_run_path(output_path / "mooseherder_ex02d_solve"),
)

print(f"Edited copy: {edited_input}")
print(result.output_file)
