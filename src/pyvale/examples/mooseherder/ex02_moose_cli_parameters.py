# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""MOOSE command-line parameters
=============================

Override case00_HEX8's named variables without modifying its source deck.
This 3D thermo-mechanical cube uses m, Pa, seconds and degrees Celsius. Native
MOOSE block paths also work. Here the thermal/mechanical history runs for five
seconds with one-second steps; endTime also sets the prescribed loading rate.
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
parameters: dict[str, mh.MooseValue] = {
    "EMod": 110e9,
    "nElemX": 3,
    "endTime": 5,
    "timeStep": 1,
    "Executioner/nl_max_its": 25,
}
result: mh.RunResult = mh.MooseRunner().run(
    input_path,
    parameters=parameters,
    output_dir=mh.next_run_path(output_path / "ex02_solve"),
)
print(result.command)
print(result.log.read_text(encoding="utf-8")[-500:])
