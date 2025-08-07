# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Running Gmsh once
================================================================================

In this example we will run a gmsh script to generate a mesh file using the
GmshRunner class.

**Installing gmsh**: For this example you will need to have a gmsh executable
which can be downloaded and installed from here: https://gmsh.info/#Download

We start by importing what we need for this example.
"""

import time
from pathlib import Path

#pyvale imports
import pyvale.dataset as dataset
from pyvale.mooseherder import GmshRunner

#%%
# First we need to create a 'runner' for gmsh which needs to know the path to
# the gmsh executable. You will need to replace this path with the path to where
# you have install gmsh on your system.
gmsh_path = Path.home() / 'gmsh/bin/gmsh'
gmsh_runner = GmshRunner(gmsh_path)

#%%
# Next we grab a gmsh file from pyvale simulation library and we set this as the
# input file for our runner.
gmsh_input = dataset.sim_case_gmsh_file_path(case_num=17)
gmsh_runner.set_input_file(gmsh_input)

#%%
# Now we can run gmsh to generate our mesh using the run method, the parse only
# flag means we will run gmsh head less and not open the gmsh GUI but terminal
# output will still be written to stdout.
#
# We also use our performance timer to time how long our mesh generation takes
# and then we print this to the console. Note that parallelisation options for
# gmsh can be controlled in the gmsh .geo script file.
start_time = time.perf_counter()
gmsh_runner.run(gmsh_input,parse_only=True)
run_time = time.perf_counter() - start_time

print()
print("-"*80)
print(f'Gmsh run time = {run_time :.3f} seconds')
print("-"*80)
print()

#%%
# The GmshRunner and the MooseRunner implement the SimRunner abstract base
# class. Later on when we will see that we can use this to run a list of
# different sim runners in order using MooseHerd workflow manager. This allows
# us to first build our mesh with gmsh and then run a moose simulation using
# that mesh. We can also implement our own SimRunner's to add additional pre or
# post processing steps to our simulation chain.