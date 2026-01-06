# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Read parameter sweep results for a MOOSE simulation
================================================================================

In this example we read the all the simulation results from multiple calls to
her workflow manager and verify that we correctly read all of the simulation
outputs.

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
import pyvale.dataset as dataset
from pyvale.mooseherder import (MooseHerd,
                                MooseRunner,
                                MooseConfig,
                                InputModifier,
                                DirectoryManager,
                                SweepLoader,
                                sweep_param_grid)


#%%
# The first part of this example is the same as our previous example called:
# 'Using multiple calls to run parallel sweeps'. For a detailed explanation of
# the code below head to that example. For now we use this to generate multiple
# sets of outputs and then use a sweep reader to read this all in below.

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
                "PRatio":(0.3,)}
params = [moose_params,]
sweep_params = sweep_param_grid(params)

print("\nParameter sweep variables by simulation:")
for ii,pp in enumerate(sweep_params):
    print(f"Sim: {ii}, Params [moose,]: {pp}")

num_para_runs: int = 3
if __name__ == '__main__':
    sweep_times = np.zeros((num_para_runs,),dtype=np.float64)
    for rr in range(num_para_runs):
        herd.run_para(sweep_params)
        sweep_times[rr] = herd.get_sweep_time()

print()
for ii,ss in enumerate(sweep_times):
    print(f"Sweep {ii} took: {ss:.3f}seconds")
print()

#%%
# Here we will just read all the results sequentially and verify that we have
# the number of simulation results we expect for the multiple calls above.
# Looking at our parameter sweep we have 4 unique combination of variables and
# we ran this 3 times so we expect to have a total of 12 simulation results in
# our outer list which we print below. Note that our inner list still has a
# single ``SimData`` object as we only have a single moose simulation in our
# simulation chain.
sweep_reader = SweepLoader(dir_manager)
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

print(f'Read time (sequential) = {read_time_seq:.6f} seconds')

