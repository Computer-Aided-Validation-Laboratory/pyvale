# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Reading exodus output from a MOOSE simulation
================================================================================

In this example we ...

**Installing moose**: To run this example you will need to have installed moose
on your system. As moose supports unix operating systems windows users will need
to use windows subsystem for linux (WSL). We use the proteus moose build which
can be found here: https://github.com/aurora-multiphysics/proteus. Build scripts
for common linux distributions can be found in the "scripts" directory of the
repo. You can also create your own moose build using instructions here:
https://mooseframework.inl.gov/.

We start by importing what we need for this example.
"""

import time
import shutil
from pathlib import Path
from typing import Any
import dataclasses
import numpy as np

#pyvale imports
import pyvale.dataset as dataset
import pyvale.sensorsim as sens
from pyvale.mooseherder import (MooseRunner,
                                MooseConfig,
                                ExodusLoader)

#%%
# We also define a helper function that will print all attriubutes of a
# dataclas so we can see what it contains. This will be useful when we inspect
# what our ``SimData`` objects contain.
def print_attrs(in_obj: Any) -> None:
    for field in dataclasses.fields(in_obj):
        if not field.name.startswith('__'):
            print(f"    {field.name}: {field.type}")

#%%
# We need to know where our simulation output is so we are going to create our
# standard pyvale-output directory, grab our simulation input file from the
# pyvale simulation library and then copy it to this directory to run. This
# means the output exodus will appear in the same directory as the input file.

output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

moose_file = dataset.element_case_input_path(dataset.EElemTest.HEX20)
moose_input = output_path / moose_file.name

shutil.copyfile(moose_file,moose_input)


#%%
# We now create our moose runner with the same method we have used in previous
# examples. We run the simulation and time it, printing the solve time to the
# terminal.

config = {"main_path": Path.home()/ "moose",
          "app_path": Path.home() / "proteus",
          "app_name": "proteus-opt"}
moose_config = MooseConfig(config)

moose_runner = MooseRunner(moose_config)

moose_runner.set_run_opts(n_tasks=1, n_threads=4, redirect_out=True)

moose_runner.set_input_file(moose_input)

start_time = time.perf_counter()
moose_runner.run()
run_time = time.perf_counter() - start_time

print("-"*80)
print(f"MOOSE run time = {run_time:.3f} seconds")
print("-"*80)

#%%
# Now we create our exodus reader by giving it the path to the exodus file we
# want to read. By default moose creates an exodus output with the input file
# name with "_out.e" appended.
output_exodus = output_path / (moose_input.stem + "_out.e")
exodus_reader = ExodusLoader(output_exodus)

print("\nReading exodus file with ExodusReader:")
print(output_exodus.resolve())
print()


#%%
# We start with the simplest method which is to just read everything in the
# exodus file and return it as a ``SimData`` object. In some cases we will not
# want to read everything into memory so we will show how we can control this n
# next.
#
# We then use a helper function to print the sim data fields to the terminal so
# we can see the structure of the dataclass. The documentation for the
# ``SimData`` class provides descriptions of each of the fields and we
# recommend you check this out to understand the terminal output.
all_sim_data = exodus_reader.load_all_sim_data()
print("SimData from 'read_all':")
sens.simtools.print_sim_data(all_sim_data)

#%%
# We are now going to read specific variables from the exodus output using a
# read configuration object. There are two ways to create this object. A good
# way to start is to use the exodus reader to return the read config that would
# extract all variables from the exodus as shown below. This is helpful as it
# will pre-populate the 'node', 'elem' and 'glob' variables with the appropriate
# dicitionary keys to read based on what is already in the exodus file.

read_config = exodus_reader.get_read_config()
sens.simtools.print_dataclass_fields(read_config)

#%%
# We set the 'node_vars' field to None to prevent the nodal variables being read
# from the exodus file. We then use the read function to return a ``SimData``
# object and we print the 'node_vars' field to verify that it has not been read.
read_config.node_vars = None
sim_data = exodus_reader.load_sim_data(read_config)

print("Read config without 'node_vars':")
print(f"    {sim_data.node_vars=}")
print()

#%%
# We can also turn off reading of the simulation time steps, nodal coordinates
# and the connectivity table by setting these flags to False in our read config.
read_config.time = False
read_config.coords = False
read_config.connect = False
sim_data = exodus_reader.load_sim_data(read_config)

print("Read config without time, coords and connectivity:")
print(f"    {sim_data.time=}")
print(f"    {sim_data.coords=}")
print(f"    {sim_data.connect=}")
print()


#%%
# We can also read specific keyed fields from 'node', 'elem' and 'glob'
# variables. Here we will read just the x displacement from the node variables.
# Note that for element variables you also need to specify the block number (
# corresponding to the number X in the key for the connectivity table in the
# format "connectivityX" in the connectivity dictionary).

read_config.node_vars = ("disp_x",)
sim_data = exodus_reader.load_sim_data(read_config)
print("Read config only extracting x displacement:")
print(f"    {sim_data.node_vars.keys()=}")
print(f"    {sim_data.node_vars['disp_x'].shape=}")
print()

