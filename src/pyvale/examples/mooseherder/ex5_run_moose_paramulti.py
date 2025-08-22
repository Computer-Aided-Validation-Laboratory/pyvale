# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Using multiple calls to run parallel sweeps
================================================================================

In this example we demonstrate how multiple repeated calls can be made to run
'herd' workflow manager where the simulations do not overwrite each other,
instead they accumulate within the output directories. If you need the
simulation output to be cleared after each call to run a sweep sequentially or
in parallel then you will need to call clear using the directory manager.

**Installing moose**: To run this example you will need to have installed moose
on your system. As moose supports unix operating systems windows users will need
to use windows subsystem for linux (WSL). We use the proteus moose build which
can be found here: https://github.com/aurora-multiphysics/proteus. Build scripts
for common linux distributions can be found in the 'scripts' directory of the
repo. You can also create your own moose build using instructions here:
https://mooseframework.inl.gov/.

We start by importing what we need for this example. For this example the
everything at the start is similar to previous examples where we have setup
our herd workflow manager. So, if you feel confident with things so far then
skip down to the last section.
"""

from pathlib import Path
import numpy as np

#pyvale imports
import pyvale.dataset as dataset
from pyvale.mooseherder import (MooseHerd,
                                MooseRunner,
                                MooseConfig,
                                InputModifier,
                                DirectoryManager,
                                sweep_param_grid)

#%%
# First we setup an input modifier and runner for our moose simulation in
# exactly the same way as we have done in previous examples.

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

#%%
# We use the moose input modifier and runner to create our herd workflow manager
# as we have seen in previous examples.
num_para_sims: int = 4
dir_manager = DirectoryManager(n_dirs=num_para_sims)
herd = MooseHerd([moose_runner],[moose_modifier],dir_manager)
herd.set_num_para_sims(n_para=num_para_sims)

#%%
# We need somewhere to run our simulations and store the output so we create our
# standard pyvale output directory as we have done in previous examples and then
# pass this to our directory manager.
output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

dir_manager.set_base_dir(output_path)
dir_manager.reset_dirs()

#%%
# We generate a grid sweep of the variables we are interested in analysing as
# we have done previously and then print this to the console so we can check
# all combinations of variables that we want are present and that the total
# number of simulations makes sense.

moose_params = {"nElemX": (2,3),
                "lengX": np.array([10e-3,15e-3]),
                "PRatio":(0.3,)}
params = [moose_params,]
sweep_params = sweep_param_grid(params)

print("\nParameter sweep variables by simulation:")
for ii,pp in enumerate(sweep_params):
    print(f"Sim: {ii}, Params [moose,]: {pp}")

print()
print(f"Total simulations = {len(sweep_params)}")
print()

#%%
# Here we are going to run the parameter sweep a certain number of times and
# while storing the total time to complete the parameter sweep each time. Once
# we have completed all the parameter sweeps we print the time taken for each
# sweep and the average sweep time to the console.
#
# Now if we inspect the simulation working directories in our pyvale-output
# directory we will see that all runs have been stored. If we need to clear
# the directories in between parallel sweeps we can call
# ``dir_manager.reset_dirs()`` and then we will only be left with one copy of
# the sweep output. Retaining all simulations is useful if we want to update
# the parameters we are passing to the ``run_para`` function every time it
# is called.

num_para_runs: int = 3

if __name__ == '__main__':
    sweep_times = np.zeros((num_para_runs,),dtype=np.float64)
    for rr in range(num_para_runs):
        herd.run_para(sweep_params)
        sweep_times[rr] = herd.get_sweep_time()


print(80*"-")
for ii,ss in enumerate(sweep_times):
    print(f"Sweep {ii} took: {ss:.3f}seconds")

print(80*"-")
print(f"Average sweep time: {np.mean(sweep_times):.3f} seconds")
print(80*"-")








