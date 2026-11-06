# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""MOOSE command-line parameters
=============================

Override named variables without modifying a deck. Parameters may also use
native slash-separated MOOSE block paths. A vector is represented by a Python
list or tuple; users do not supply shell quotes.
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
# Root variables remain referenced by the included physics and solver blocks.
parameters: dict[str, mh.MooseValue] = {
    "EMod": 110e9,
    "topDisp": 2e-4,
    "nElemX": 4,
    "Executioner/nl_max_its": 25,
}
result: mh.RunResult = mh.MooseRunner().run(
    data_path / "model.i",
    parameters=parameters,
    output_dir=mh.next_run_path(output_path / "ex02_solve"),
)
print(result.command)
print(result.log.read_text(encoding="utf-8")[-500:])
