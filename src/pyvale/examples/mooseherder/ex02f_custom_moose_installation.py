# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Select a custom MOOSE installation
===================================

Use an explicit MOOSE executable and OS environment variable overrides to solve
case00_HEX8, the packaged thermo-mechanical cube. Leave the executable as None
to use Rabbit. The cube generates its own mesh and uses m, Pa, seconds and
degrees Celsius. An invalid custom executable never falls back to Rabbit.
"""

from pathlib import Path

import pyvale.data as dataset
import pyvale.mooseherder as mh

# %%
output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

input_path: Path = dataset.element_case_input_path(dataset.EElemTest.HEX8)
print(f"Packaged MOOSE input: {input_path}")

# %%
# Set this to your application executable, for example Path('/opt/moose/app-opt').
moose_exe_file: Path | None = None

# %%
# OS environment variable overrides are merged with the calling Python process's
# OS environment variables for the solver process. An empty dictionary inherits
# those OS environment variables without overrides; it does not select a venv.
env_vars: dict[str, str] = {}
result: mh.RunResult = mh.MooseRunner(
    moose_exe_file=moose_exe_file, env_vars=env_vars
).run(
    input_path,
    output_dir=mh.get_next_run_path(output_path / "ex02f_solve"),
)
print(result.output_file)
