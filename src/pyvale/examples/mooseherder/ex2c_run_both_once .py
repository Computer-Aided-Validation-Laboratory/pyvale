# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Run Gmsh then MOOSE once
================================================================================

In this example we ...
"""

#%%
#

import time
import shutil
from pathlib import Path
import pyvale as pyv
from pyvale.mooseherder import (MooseConfig,
                                GmshRunner,
                                MooseRunner)


#%%
#

output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

gmsh_file = pyv.DataSet.sim_case_gmsh_file_path(case_num=17)
gmsh_input = output_path / gmsh_file.name

moose_file = pyv.DataSet.sim_case_input_file_path(case_num=17)
moose_input = output_path / moose_file.name

shutil.copyfile(moose_file,moose_input)
shutil.copyfile(gmsh_file,gmsh_input)

#%%
#
gmsh_path = Path.home() / 'gmsh/bin/gmsh'
gmsh_runner = GmshRunner(gmsh_path)

gmsh_runner.set_input_file(gmsh_input)

gmsh_start = time.perf_counter()
gmsh_runner.run()
gmsh_run_time = time.perf_counter()-gmsh_start

config_path = Path.cwd() / 'moose-config.json'

#%%
#
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


print("-"*80)
print(f'Gmsh run time = {gmsh_run_time:.2f} seconds')
print(f'MOOOSE run time = {moose_run_time:.2f} seconds')
print("-"*80)
print()


