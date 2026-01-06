# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Reading exodus output from a parameter sweep
================================================================================

In this example we run a parallel sweep of a moose simulation and then read the
results of the whole sweep using the sweep reader class.

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
import numpy as np

#pyvale imports
import pyvale.sensorsim as sens
import pyvale.dataset as dataset
from pyvale.mooseherder import (MooseHerd,
                                MooseRunner,
                                MooseConfig,
                                InputModifier,
                                DirectoryManager,
                                SweepLoader,
                                sweep_param_grid)

#%%
# In this first section we setup our herd workflow manager to run a parameter
# sweep of our moose simulation as we have done in previous examples. We run
# the parameter sweep and print the solve time to the terminal. The sweep
# output is in the standard pyvale-output directory we have used previously.
# In the next section we will read the output from the parameter sweep below.

moose_input = dataset.element_case_input_path(dataset.EElemTest.HEX20)
moose_modifier = InputModifier(moose_input,'#','')

config = {'main_path': Path.home()/ 'moose',
          'app_path': Path.home() / 'proteus',
          'app_name': 'proteus-opt'}
moose_config = MooseConfig(config)

moose_runner = MooseRunner(moose_config)
moose_runner.set_run_opts(n_tasks = 1,
                          n_threads = 2,
                          redirect_out = True)

num_para_sims: int = 4
dir_manager = DirectoryManager(n_dirs=num_para_sims)
herd = MooseHerd([moose_runner],[moose_modifier],dir_manager)
herd.set_num_para_sims(n_para=num_para_sims)

output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

dir_manager.set_base_dir(output_path)
dir_manager.reset_dirs()

moose_params = {"nElemX": (2,3),
                "lengX": np.array([10e-3,15e-3]),
                "PRatio":(0.3,0.35)}
params = [moose_params,]
sweep_params = sweep_param_grid(params)


if __name__ == "__main__":
    print('Running simulation parameter sweep in parallel.')
    herd.run_para(sweep_params)
    print(f'Run time (parallel) = {herd.get_sweep_time():.3f} seconds\n')


#%%
# To read the sweep output files we first create our sweep reader and pass it
# the same directory manager we used to run the sweep. We also set the number
# of simulation outputs to read in parallel when we call the read parallel
# function. We will see below that we can still read sequentially by calling
# read sequential functions and if the simulation output files are small it is
# likely to be faster to read them sequentially.
#
# We first use our sweep reader to inspect the output path keys to find the
# simulation output files that exist in the simulation working directories.

sweep_reader = SweepLoader(dir_manager,num_para_read=4)
output_files = sweep_reader.read_all_output_file_keys()

print('Sweep output files (from output_keys.json):')
for ff in output_files:
    print(f"    {ff}")
print()

#%%
# Using the sweep reader we can read the results for a single simulation chain
# from the sweep. Our simulation chain only has a single moose simulation so
# the list of ``SimData`` objects we are returned only has a single element.
# We then use a helper function to print the contents of the ``SimData`` object
# to the terminal.
#
# We suggest you check out the documentation for the ``SimData`` object as it
# includes a detailed description of each of the relevant fields you might want
# to use for post-processing.
sim_data_list = sweep_reader.read_results_once(output_files[0])
sens.simtools.print_sim_data(sim_data_list[0])

#%%
# We can use the sweep reader to read results for each simulation chain in the
# sweep sequentially with read sequential function. The sweep results we are
# returned is a list of list of data classes where the outer list corresponds to
# the unique simulation chain in the sweep and the inner list corresponds to the
# the results for the particular simulation tool in the chain.
#
# After reading the sweep results we print the inner and outer list lengths. We
# have 8 unique simulation chains with a single simulation tool (moose) in the
# chain.
start_time = time.perf_counter()
sweep_results_seq = sweep_reader.read_sequential()
read_time_seq = time.perf_counter() - start_time

print("Outer list = unique simulation chain:")
print(f"    {len(sweep_results_seq)=}")
print("Inner list = particular simulation tool in the chain:")
print(f"    {len(sweep_results_seq[0])=}")
print("'SimData' object for the particular simulation tool:")
print(f"    {type(sweep_results_seq[0][0])=}")
print()

#%%
# Finally, we read the same sweep in parallel making sure we include a main
# guard as we will be using the multi-processing package to do this. We then
# print the read time to the console for the sequential and parallel reads.
if __name__ == '__main__':
    start_time = time.perf_counter()
    sweep_results_para = sweep_reader.read_results_para()
    read_time_para = time.perf_counter() - start_time

print()
print("-"*80)
print(f'Read time sequential = {read_time_seq:.6f} seconds')
print(f'Read time parallel   = {read_time_para:.6f} seconds')
print("-"*80)
print()
