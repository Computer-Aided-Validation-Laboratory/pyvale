# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Running MOOSE once
================================================================================

In this example we will run a single moose simulation from a moose input .i file
using a 'runner' object.

**Installing moose**: To run this example you will need to have installed moose
on your system. As moose supports unix operating systems windows users will need
to use windows subsystem for linux (WSL). We use the proteus moose build which
can be found here: https://github.com/aurora-multiphysics/proteus. Build scripts
for common linux distributions can be found in the 'scripts' directory of the
repo. You can also create your own moose build using instructions here:
https://mooseframework.inl.gov/.

We start by importing what we need for this example.
"""

import time
from pathlib import Path

#pyvale imports
import pyvale.dataset as dataset
from pyvale.mooseherder import (MooseConfig,
                                MooseRunner)


#%%
# First we build our moose configuration which gives the location of our main
# moose build, our moose app and the name of the app to use when called on the
# command line.
config = {'main_path': Path.home()/ 'moose',
          'app_path': Path.home() / 'proteus',
          'app_name': 'proteus-opt'}
moose_config = MooseConfig(config)

#%%
# We can now build a runner object using our configuration. We can then set some
# options for the run including parallelisation and if we should redirect
# terminal output to file. For smaller simulations we are better off using
# threads for paralleisation as they reduce overhead compared to MPI tasks.
moose_runner = MooseRunner(moose_config)

moose_runner.set_run_opts(n_tasks = 1,
                          n_threads = 8,
                          redirect_out = False)

#%%
# Let's grab a simple thermo-mechanical cube test case from pyvale's moose
# simulation library and we will set this as the input file to run with our
# 'runner'.
moose_input = dataset.element_case_input_path(dataset.EElemTest.HEX20)
moose_runner.set_input_file(moose_input)

#%%
# Our moose runner will pass a list of strings which form the command line to
# run our moose simulation. We print the list of command line arguments here so
# we can check we are correctly calling our input file with the run options we
# want.
print(moose_runner.get_arg_list())
print()

#%%
# To run our moose simulation we just need to call 'run', here we will time our
# moose run and then print the solve time to the terminal
start_time = time.perf_counter()
moose_runner.run()
run_time = time.perf_counter() - start_time

print()
print("-"*80)
print(f'MOOSE run time = {run_time:.3f} seconds')
print("-"*80)
print()
