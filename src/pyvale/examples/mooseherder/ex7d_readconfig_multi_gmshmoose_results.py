# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Using read config to extract specific variables from a sweep
================================================================================

In this example we are going to use a read configuration object to control what
variables are read from our simulation output. This is useful for when we might
only want to read a subset of the simulation data for post-processing. For
example a post processor like 'max_temp' from 'glob_vars' without reading in the
full temperature field. This can help save memory when reading in data from
large sweeps.

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
                                ExodusLoader,
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
# Now we create a sweep reader as we have done before and we will use this to
# read the json keys containing the paths to all the output files. The output
# file paths have the same list of lists structure we get when we read the
# sweep data where the outer list corresponds to the unique simulation chain and
# the inner list corresponds to position of the specific simulation tool in the
# chain.
#

sweep_reader = SweepLoader(dir_manager,num_para_read=4)
output_file_paths = sweep_reader.read_all_output_file_keys()


#%%
# When we used a read configuration with our exodus reader previously we saw
# that it can be simpler to get the read configuration that covers everything
# in our file and then edit it to extract what we want. Here we are use our
# exodus reader to extract the read configuration fo the first simulation.
# We also get the original sim data so we can compare to the case when we use
# a read configuration to control what we read.
#
# Now we want to edit the read configuration so that we only read every second
# time step from our simulation output files. We do this by getting the
# original time steps with our exodus reader and then using this to change the
# ``time_inds`` field of our read configuration to extract every second step.
exodus_reader = ExodusLoader(output_file_paths[0][0])
sim_data_orig = exodus_reader.load_all_sim_data()

read_config = exodus_reader.get_read_config()
sim_time = exodus_reader.get_time()
read_config.time_inds = np.arange(0,sim_time.shape[0],2)

#%%
# Now we read the sweep results using our read configuration and then check
# extracted simulation data object to see that we have every second time step.
sweep_results_seq = sweep_reader.read_sequential(read_config=read_config)

print("Comparison of time steps extracted:")
print()
print(f"{sim_data_orig.time.shape=}")
print(f"{sweep_results_seq[0][0].time.shape=}")
print()
print(f"{sim_data_orig.node_vars['disp_x'].shape=}")
print(f"{sweep_results_seq[0][0].node_vars['disp_x'].shape=}")
print()

