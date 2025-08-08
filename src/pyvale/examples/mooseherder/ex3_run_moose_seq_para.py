# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Running a parameter sweep of a MOOSE simulation
================================================================================

In this example we will perform a parameter sweep of a moose simulation showing
the capability of the 'herder' workflow manager which can be passed a list of
'input modifiers' and 'runners'. The 'herder' will then use the 'input
modifiers' to update simulation parameters and then call the respective 'runner'
using the modified input file. In this example we will also see that the
'herder' can be used to execute a parameter sweep sequentially or in parallel.

**Installing moose**: To run this example you will need to have installed moose
on your system. As moose supports unix operating systems windows users will need
to use windows subsystem for linux (WSL). We use the proteus moose build which
can be found here: https://github.com/aurora-multiphysics/proteus. Build scripts
for common linux distributions can be found in the 'scripts' directory of the
repo. You can also create your own moose build using instructions here:
https://mooseframework.inl.gov/.

We start by importing what we need for this example.
"""

from pathlib import Path
import numpy as np

#pyvale imports
import pyvale.dataset as dataset
from pyvale.mooseherder import (MooseHerd,
                                MooseRunner,
                                InputModifier,
                                DirectoryManager,
                                MooseConfig,
                                sweep_param_grid)

#%%
# First we are going to setup an input modifier and a runner for our moose
# simulation. Here we need to make sure that when we set our moose
# parallelisation options we leave enough threads for all the simulations that
# are running at once base on our CPU. It is also helpful to redirect stdout to
# file so that our terminal does not become a mess when we start running our
# simulations in parallel.
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
# Now we are going to create a directory manager which will be used to make sure
# our simulations are run in separate directories. We then create our herd
# workflow manager with our list of runners and corresponding input modifiers.
# In our case we are only running moose so our lists have a single item. The
# last thing we do is specify the number of simulations we want to run in
# parallel, for this case we match the number of directories.
num_para_sims: int = 4
dir_manager = DirectoryManager(n_dirs=num_para_sims)
herd = MooseHerd([moose_runner],[moose_modifier],dir_manager)
herd.set_num_para_sims(n_para=num_para_sims)

#%%
# We need somewhere to run our simulations and store the output so we create our
# standard pyvale output directory and then we set this as the base directory
# for our directory manager. We clear any old output directories and then create
# new ones ready to write our simulation output to.
output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

dir_manager.set_base_dir(output_path)
dir_manager.reset_dirs()

#%%
# We now need to generate the parameter combinations we want to run using our
# 'herd'. This is given as a list of list of dictionaries where the outer list
# corresponds to the unique simulation chain, the inner list corresponds to each
# simulation runner in the chain, and the dicitionary contains key value pairs
# where the keys are the variables names to edit in the input file. For this
# case we only have moose in our simulation chain so our inner list will only
# have a length of one but in the next example we will see how we can combine
# a parameter sweep with gmsh->moose sweeping all possible combinations of
# variables for both simulation tools.
#
# For now we are going to use a helper function from mooseherder which will
# generate all possible combination for us in the correct data format. We just
# provide a dictionary of lists of unique parameters we want to analyse. Finally
# we print the unique combinations of parameters to the terminal as well as the
# total number of simulations to check everything is working as expected.

moose_params = {"nElemX": (2,3,4),
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
# The run once function of the herd allows us to run a particular single
# simulation chain from anywhere in the sweep. This is useful for debugging when
# you want to rerun a single case to see the output or what went wrong. The herd
# also stores the solution time for each single iteration so we will store this
# to estimate how long the whole sweep should take when solving sequentially.
herd.run_once(0,sweep_params[0])
time_run_once = herd.get_iter_time()


#%%
# We can run the whole parameter sweep sequentially (one by one) using the run
# sequential function of the herd. We also store the total solution time
# for all simulation chains so that we can compare to a parallel run later. Note
# that it can be beneficial to run sequentially if you are using the herd within
# another loop or if one of the steps in your simulation chain is expensive and
# that step needs the computational resource.
herd.run_sequential(sweep_params)
time_run_seq = herd.get_sweep_time()

#%%
# Finally, we can run our parameter sweep in parallel. We need a main guard here
# as we use the multi-processing package. We also store the sweep time for this
# case to compare our sequential to parallel run time.
if __name__ == "__main__":
    herd.run_para(sweep_params)
    time_run_para = herd.get_sweep_time()

#%%
# Now that we have run all cases we can compare run times for a single
# simulation multiplied by the total number of simulations against runnning the
# sweep in parallel
print("-"*80)
print(f'Run time (one iter)             = {time_run_once:.3f} seconds')
print(f'Est. time (one iter x num sims) = {(time_run_once*len(sweep_params)):.3f} seconds')
print()
print(f'Run time (seq)      = {time_run_seq:.3f} seconds')
print(f'Run time (para)     = {time_run_para:.3f} seconds')
print("-"*80)
print()


