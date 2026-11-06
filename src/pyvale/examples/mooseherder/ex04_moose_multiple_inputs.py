# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Ordered MOOSE entry decks
=========================

MOOSE merges multiple entry decks in their supplied order. Later decks override
earlier ones, then CLI parameters take precedence. This is distinct from an
entry deck resolving its own !include dependencies.
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
inputs: tuple[Path, ...] = (
    data_path / "model.i",
    data_path / "case_overlay.i",
)
result: mh.RunResult = mh.MooseRunner().run(
    inputs,
    input_root=data_path,
    parameters={"topDisp": 2e-4},
    output_dir=mh.next_run_path(output_path / "ex04_solve"),
)
print(result.command)
