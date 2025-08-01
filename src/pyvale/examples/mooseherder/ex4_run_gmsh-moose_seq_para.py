# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Running a parameter sweep of a Gmsh and MOOSE simulation
================================================================================

In this example we will perform a parameter sweep of a gmsh-moose simulation
chain to demonstrate the capability of the 'herder' workflow manager. Here we
pass the 'herder' a list of simulation tools that we want to modify inputs for
(i.e. input modifiers) and then run (i.e. with runners). The simulation tools
are called sequentially in the order they are in the lists so we will need to
make sure we call gmsh first to generate the mesh and then call moose to use the
mesh to run the simulation.

As in the previous example we will generate a parameter sweep and then run it
sequentially and in parallel and compare the run times.

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

from pathlib import Path
import numpy as np
import pyvale as pyv
from pyvale.mooseherder import (MooseHerd,
                                MooseRunner,
                                MooseConfig,
                                GmshRunner,
                                InputModifier,
                                DirectoryManager,
                                sweep_param_grid)

#%%
# First we setup our input modifer and runner for gmsh using the same 2D plate
# with a hole simulation test case from the pyvale simulation library. This is
# the same as we have seen in previous examples for the gmsh input modifier and
# running gmsh.
sim_case: int = 17

gmsh_input = pyv.DataSet.sim_case_gmsh_file_path(case_num=sim_case)
gmsh_modifier = InputModifier(gmsh_input,"//",";")

gmsh_path = Path.home() / "gmsh/bin/gmsh"
gmsh_runner = GmshRunner(gmsh_path)
gmsh_runner.set_input_file(gmsh_input)


#%%
# Next we setup our moose input modifier and runner in the same way as we have
# done in previous examples. We set our parallelisation options for moose here
# as well as redirecting stdout to file to save our terminal when we run in
# parallel.
moose_input = pyv.DataSet.sim_case_input_file_path(case_num=sim_case)
moose_modifier = InputModifier(moose_input,"#","")

config = {'main_path': Path.home()/ 'moose',
          'app_path': Path.home() / 'proteus',
          'app_name': 'proteus-opt'}
moose_config = MooseConfig(config)
moose_runner = MooseRunner(moose_config)
moose_runner.set_run_opts(n_tasks = 1,
                          n_threads = 2,
                          redirect_out = True)

#%%
# We can now setup our 'herd' workflow manager making sure me place gmsh ahead
# of moose in the input modifier and runner lists so it is executed first to
# generate our mesh. We setup our directories and number of simulations to run
# in paralle as we have done previously.
num_para_sims: int = 4

sim_runners = [gmsh_runner,moose_runner]
input_modifiers = [gmsh_modifier,moose_modifier]
dir_manager = DirectoryManager(n_dirs=num_para_sims)

herd = MooseHerd(sim_runners,input_modifiers,dir_manager)
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
dir_manager.clear_dirs()
dir_manager.create_dirs()


gmsh_params = {"plate_width": np.array([150e-3,100e-3]),
               "plate_height": "plate_width + 100e-3"}
moose_params = None
params = [gmsh_params,moose_params]
sweep_params = sweep_param_grid(params)

print("\nParameter sweep variables by simulation:")
for ii,pp in enumerate(sweep_params):
    print(f"Sim: {ii}, Params [gmsh,moose]: {pp}")






# #%%
# # The run once function of the herd allows us to run a particular single
# # simulation chain from anywhere in the sweep. This is useful for debugging when
# # you want to rerun a single case to see the output or what went wrong. The herd
# # also stores the solution time for each single iteration so we will store this
# # to estimate how long the whole sweep should take when solving sequentially.
# herd.run_once(0,sweep_params[0])
# time_run_once = herd.get_iter_time()


# #%%
# # We can run the whole parameter sweep sequentially (one by one) using the run
# # sequential function of the herd. We also store the total solution time
# # for all simulation chains so that we can compare to a parallel run later. Note
# # that it can be beneficial to run sequentially if you are using the herd within
# # another loop or if one of the steps in your simulation chain is expensive and
# # that step needs the computational resource.
# herd.run_sequential(sweep_params)
# time_run_seq = herd.get_sweep_time()

# #%%
# # Finally, we can run our parameter sweep in parallel a
# if __name__ == "__main__":
#     herd.run_para(sweep_params)
#     time_run_para = herd.get_sweep_time()

# #%%
# # Now that we have run all cases we can compare run times for a single
# # simulation multiplied by the total number of simulations against runnning the
# # sweep in parallel
# print("-"*80)
# print(f'Run time (one iter)             = {time_run_once:.3f} seconds')
# print(f'Est. time (one iter x num sims) = {(time_run_once*len(sweep_params)):.3f} seconds')
# print()
# print(f'Run time (seq)      = {time_run_seq:.3f} seconds')
# print(f'Run time (para)     = {time_run_para:.3f} seconds')
# print("-"*80)
# print()
