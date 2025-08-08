# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Run Gmsh then MOOSE once
================================================================================

In this example we use a gmsh runner followed by a moose runner to generate our
mesh and then run a moose simulation using this mesh. The moose input file needs
to know the name of the gmsh .msh file which is specified in the gmsh .geo
script when the Save command is called. It is possible to use the input
modifiers we have seen previously to update this file name as a variable in the
moose input script but for this example we have set things manually inside the
moose input script.

**Installing moose**: To run this example you will need to have installed moose
on your system. As moose supports unix operating systems windows users will need
to use windows subsystem for linux (WSL). We use the proteus moose build which
can be found here: https://github.com/aurora-multiphysics/proteus. Build scripts
for common linux distributions can be found in the 'scripts' directory of the
repo. You can also create your own moose build using instructions here:
https://mooseframework.inl.gov/.

**Installing gmsh**: For this example you will need to have a gmsh executable
which can be downloaded and installed from here: https://gmsh.info/#Download

We start by importing what we need for this example.
"""

import time
import shutil
from pathlib import Path

#pyvale imports
import pyvale.dataset as dataset
from pyvale.mooseherder import (MooseConfig,
                                GmshRunner,
                                MooseRunner)

#%%
# We need to make sure the output .msh file from gmsh can be found by our moose
# input script so we are going to put them in our standard pyvale-output
# directory in our current working directory. First we grab the paths for the
# .geo and .i and then we copy them to the pyvale-output directory where we will
# run our simulation from. We then print the paths to we can see where the files
# are - try opening them with your text editor of choice so you can see how the
# name of the mesh is specified in the gmsh .geo as the .msh output and the then
# how the name is matched in the moose .i to read the .msh to run the sim.

output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

gmsh_file = dataset.sim_case_gmsh_file_path(case_num=17)
gmsh_input = output_path / gmsh_file.name

moose_file = dataset.sim_case_input_file_path(case_num=17)
moose_input = output_path / moose_file.name

shutil.copyfile(moose_file,moose_input)
shutil.copyfile(gmsh_file,gmsh_input)

print(f"\n{moose_input.resolve()=}")
print(f"{gmsh_input.resolve()=}\n")

#%%
# We need to run gmsh first to generate our .msh file so we set it up and run it
# in exactly the same way as we have done in the previous example. We pass the
# path to the gmsh executable to our runner. We then set our input file and call
# run to generate the mesh.
gmsh_path = Path.home() / 'gmsh/bin/gmsh'
gmsh_runner = GmshRunner(gmsh_path)

gmsh_runner.set_input_file(gmsh_input)

gmsh_start = time.perf_counter()
gmsh_runner.run(parse_only=True)
gmsh_run_time = time.perf_counter()-gmsh_start

#%%
# Now that we have our mesh we can run our moose simulation. We will setup and
# run moose in exactly the same was as in a previous example. First, we setup
# our moose configuration and pass this to our runner. We then set our run /
# parallelisation options before calling run to extecute the simulation.
config = {'main_path': Path.home()/ 'moose',
          'app_path': Path.home() / 'proteus',
          'app_name': 'proteus-opt'}
moose_config = MooseConfig(config)

moose_runner = MooseRunner(moose_config)

moose_runner.set_run_opts(n_tasks = 1,
                          n_threads = 4,
                          redirect_out = True)

moose_runner.set_input_file(moose_input)

moose_start = time.perf_counter()
moose_runner.run()
moose_run_time = time.perf_counter() - moose_start

#%%
# Finally we print the execution times of both runners and print these to the
# console.
print("-"*80)
print(f'Gmsh run time = {gmsh_run_time:.2f} seconds')
print(f'MOOOSE run time = {moose_run_time:.2f} seconds')
print("-"*80)
print()


