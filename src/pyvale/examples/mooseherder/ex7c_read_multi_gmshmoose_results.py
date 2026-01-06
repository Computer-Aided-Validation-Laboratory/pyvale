# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Read parameter sweep results for a Gmsh and MOOSE simulation
================================================================================

In this example we will read the results from a sweep that includes gmsh and
moose in the simulation chain. A key difference here in the results output is
that gmsh does not create output we want to read so the inner list of our
results list of lists will have 'None' in its 0 index and then the ``SimData``
data object from our moose simulation in the 1 index.

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
from pathlib import Path
import numpy as np

#pyvale imports
import pyvale.dataset as dataset
from pyvale.mooseherder import (MooseHerd,
                                MooseRunner,
                                GmshRunner,
                                MooseConfig,
                                InputModifier,
                                DirectoryManager,
                                SweepLoader,
                                sweep_param_grid)

#%%
# The first part of this example is the same as the previous example we have
# seen running a parallel parameter sweep for gmsh+moose using the herd workflow
# manager. If you are not familiar with the code below go back to the example
# entitled 'Running a parameter sweep of a Gmsh and MOOSE simulation'.
# Otherwise you can skip down to the next code block where we will read the
# output of the sweep and compare it to what we have seen previously.
sim_case: int = 17

gmsh_input = dataset.sim_case_gmsh_file_path(case_num=sim_case)
gmsh_modifier = InputModifier(gmsh_input,"//",";")

gmsh_runner = GmshRunner(gmsh_path=(Path.home() / "gmsh/bin/gmsh"))
gmsh_runner.set_input_file(gmsh_input)

moose_input = dataset.sim_case_input_file_path(case_num=sim_case)
moose_modifier = InputModifier(moose_input,"#","")

moose_config = MooseConfig({'main_path': Path.home()/ 'moose',
                            'app_path': Path.home() / 'proteus',
                            'app_name': 'proteus-opt'})
moose_runner = MooseRunner(moose_config)
moose_runner.set_run_opts(n_tasks = 1,
                          n_threads = 2,
                          redirect_out = True)

num_para_sims: int = 4
dir_manager = DirectoryManager(n_dirs=num_para_sims)
herd = MooseHerd([gmsh_runner,moose_runner],
                 [gmsh_modifier,moose_modifier],
                 dir_manager,
                 num_para_sims)

output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

dir_manager.set_base_dir(output_path)
dir_manager.reset_dirs()

gmsh_params = {"plate_width": np.array([150e-3,100e-3]),
               "plate_height": ("plate_width + 100e-3",
                                "plate_width + 50e-3")}
moose_params = None
sweep_params = sweep_param_grid([gmsh_params,moose_params])

if __name__ == "__main__":
    herd.run_para(sweep_params)
    time_run_para = herd.get_sweep_time()


print(f'Sweep run time (para) = {time_run_para:.3f} seconds\n')

#%%
# We now pass the directory manager we used for the sweep to our sweep reader
# and then we read in the sweep results sequentially. As we have seen before
# our sweep results are given as a list of lists where the outer list is the
# unique simulation chain in our parameter sweep and the inner list corresponds
# to each simulation tool in the chain. In this case we have gmsh and moose so
# our inner list should have a length of 2.

sweep_reader = SweepLoader(dir_manager,num_para_read=4)

start_time = time.perf_counter()
sweep_results_seq = sweep_reader.read_sequential()
read_time_seq = time.perf_counter() - start_time

print(f'Read time sequential = {read_time_seq:.6f} seconds\n')

print("Outer list = unique simulation chain:")
print(f"    {len(sweep_results_seq)=}")
print("Inner list = particular simulation tool in the chain:")
print(f"    {len(sweep_results_seq[0])=}")
print()

#%%
# As gmsh does not have any simulation output we can read as a ``SimData``
# object we see that our inner sweep results list has 'None' in the position
# corresponding to gmsh (the 0 index). We also see we have our moose output as
# a ``SimData`` object at the 1 index of the inner list.
print(f"{type(sweep_results_seq[0][0])=}")
print(f"{type(sweep_results_seq[0][1])=}")
