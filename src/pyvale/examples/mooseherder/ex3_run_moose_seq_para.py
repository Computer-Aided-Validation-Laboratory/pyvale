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
"""

from pathlib import Path
import itertools
import numpy as np
import pyvale as pyv
from pyvale.mooseherder import (MooseHerd,
                                MooseRunner,
                                InputModifier,
                                DirectoryManager,
                                MooseConfig,
                                sweep_param_grid)




#%%
#
moose_input = pyv.DataSet.element_case_input_path(pyv.EElemTest.HEX20)
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
#
dir_manager = DirectoryManager(n_dirs=4)
herd = MooseHerd([moose_runner],[moose_modifier],dir_manager)


herd.set_num_para_sims(n_para=4)


#%%
#
output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

dir_manager.set_base_dir(output_path)
dir_manager.clear_dirs()
dir_manager.create_dirs()

#%%
#
# Needs to be list[list[dict]] - outer list is simulation iteration,
# inner list is what is passed to each runner/inputmodifier

n_elem_x = [2,3,4]
leng_x = [10e-3,15e-3]
p_rat = [0.3,0.35]
sweep_vars = []
for nn in n_elem_x:
    for ll in leng_x:
        for pp in p_rat:
            sweep_vars.append([{'nElemX':nn,'lengX':ll,'PRatio':pp},])

print('Parameter sweep variables:')
for vv in sweep_vars:
    print(vv)
print()


moose_params = {"nElemX": [2,3,4],"lengX":np.array([10e-3,15e-3]),"PRatio":[0.3,0.35]}
gmsh_params = {"plate_width": [150e-3,100e-3], "plate_height": ["plate_width + 100e-3",]}
params = [gmsh_params,moose_params]

sweep_params = sweep_param_grid(params)

for ss in sweep_params:
    print(ss)





















# #%%
# #
# # Single run saved in sim-workdir-1
# herd.run_once(0,sweep_vars[0])
# time_run_once = herd.get_iter_time()


# #%%
# #
# # Run all variable combinations (8) sequentially in sim-workdir-1
# herd.run_sequential(sweep_vars)
# time_run_seq = herd.get_sweep_time()

# #%%
# #
# # Run all variable combinations across 4 MOOSE instances with two runs saved in
# # each sim-workdir
# if __name__ == "__main__":
#     herd.run_para(sweep_vars)
#     time_run_para = herd.get_sweep_time()


# #%%
# #
# print("-"*80)
# print(f'Run time (one iter)             = {time_run_once:.3f} seconds')
# print(f'Est. time (one iter x num sims) = {(time_run_once*len(sweep_vars)):.3f} seconds')
# print()
# print(f'Run time (seq)      = {time_run_seq:.3f} seconds')
# print(f'Run time (para)     = {time_run_para:.3f} seconds')
# print("-"*80)
# print()


