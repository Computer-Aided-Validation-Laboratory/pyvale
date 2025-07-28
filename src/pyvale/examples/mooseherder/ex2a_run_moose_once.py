# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Running MOOSE once
================================================================================

In this example we will run a single MOOSE simulation
"""

import time
from pathlib import Path
import pyvale as pyv
from pyvale.mooseherder import (MooseConfig,
                                MooseRunner)


#%%
#
config = {'main_path': Path.home()/ 'moose',
          'app_path': Path.home() / 'proteus',
          'app_name': 'proteus-opt'}
moose_config = MooseConfig(config)

#%%
#
moose_runner = MooseRunner(moose_config)

moose_runner.set_run_opts(n_tasks = 1,
                          n_threads = 8,
                          redirect_out = False)

#%%
#
moose_input = pyv.DataSet.element_case_input_path(pyv.EElemTest.HEX20)
moose_runner.set_input_file(moose_input)

#%%
#
print(moose_runner.get_arg_list())
print()

#%%
#
start_time = time.perf_counter()
moose_runner.run()
run_time = time.perf_counter() - start_time

print()
print("-"*80)
print(f'MOOSE run time = {run_time:.3f} seconds')
print("-"*80)
print()
