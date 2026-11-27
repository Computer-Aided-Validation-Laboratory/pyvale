# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""MOOSE: Custom installation
==========================

Use an explicit MOOSE executable and OS environment variable overrides to solve
case00_HEX8, the packaged thermo-mechanical cube. Leave the executable as None
to use Rabbit. The cube generates its own mesh and uses m, Pa, seconds and
degrees Celsius. An invalid custom executable never falls back to Rabbit.
"""

import os
from pathlib import Path

import pyvale.data as dataset
import pyvale.mooseherder as mh

# %%
# Create the standard output directory so generated files stay together
# under pyvale-output rather than alongside the packaged input files.
# Locate the packaged cube input to test the selected MOOSE executable.
output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

input_path: Path = dataset.element_case_input_path(dataset.EElemTest.HEX8)
print(f"Packaged MOOSE input: {input_path}")

# %%
# To use the Proteus setup from the original mooseherder examples, replace the
# MOOSE/Proteus paths and MPI compiler wrapper names below with your installation's
# settings, then set USE_PROTEUS=True. Leave it False to run with packaged Rabbit.
# The custom application executable is typically <proteus_dir>/proteus-opt.
# MPI process counts and launchers are demonstrated separately in ex02e.
USE_PROTEUS: bool = False
moose_exe_file: Path | None = None
env_vars: dict[str, str] = {}
if USE_PROTEUS:
    moose_dir: Path = Path.home() / "moose"  # Replace with your MOOSE source tree.
    proteus_dir: Path = Path.home() / "proteus"  # Replace with your Proteus directory.
    moose_exe_file = proteus_dir / "proteus-opt"
    env_vars = {
        "CC": "mpicc",
        "CXX": "mpicxx",
        "F90": "mpif90",
        "F77": "mpif77",
        "FC": "mpif90",
        "MOOSE_DIR": str(moose_dir),
        "PATH": os.pathsep.join((str(proteus_dir), os.environ.get("PATH", ""))),
    }
else:
    print(
        "Using Rabbit. To use Proteus, replace the paths and MPI compiler wrapper "
        "names in this example, then set USE_PROTEUS=True."
    )

# %%
# Pass OS environment variable overrides to the solver process without changing
# the calling Python process's OS environment variables. The compiler variables
# reproduce the old MOOSE build setup; they do not enable MPI execution or change
# how an existing executable was compiled. MOOSE_DIR identifies the source tree,
# and PATH retains existing entries while adding the Proteus executable directory.
# MooseRunner invokes proteus-opt directly when USE_PROTEUS=True; its command
# includes -i <staged_input> and --n-threads=1, plus the managed output settings.
result: mh.RunResult = mh.MooseRunner(
    moose_exe_file=moose_exe_file, env_vars=env_vars
).run(
    input_path,
    output_dir=mh.get_next_run_path(output_path / "mooseherder_ex02f_solve"),
)
print("Command:", result.command)
print(result.output_file)
