# =============================================================================="ctrl+p"
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Reading results from a pre-run parameter sweep
================================================================================

In this example we will read in the output of a parameter sweep performed by the
'herd' workflow manager specifically showing how we can read a sweep if we don't
have the original directory manager.

**NOTE**: this example assumes you have run one of the previous examples using
the herd workflow manager and that there is a series of output simulation
working directories in the pyvale-output directory. If not please run the
example called 'Running a parameter sweep of a MOOSE simulation' first.

We start by importing what we need for this example.
"""
import time
from pprint import pprint
from pathlib import Path
from pyvale.mooseherder import DirectoryManager
from pyvale.mooseherder import SweepLoader

#%%
# First we create a directory manager and pass it our standard pyvale output
# directory where our simulation sweep output is. The number of directories here
# is not critical as long as it is equal to or larger than the number of working
# directories you used when you ran your sweep it will find all simulation
# outputs that exist in the working directories.
output_base_path = Path.cwd() / "pyvale-output"
dir_manager = DirectoryManager(n_dirs=4)
dir_manager.set_base_dir(output_base_path)

#%%
# We now pass the directory manager into a sweep reader setting the option to
# read our sweep output in parallel. Note that for small output files reading
# the output in parallel will probably be slower than reading sequentially so
# make sure you test the reader for your particular case.
#
# We then use the reader extract the list of list of output file paths and the
# combinations of variables that were used for this parameter sweep.
sweep_reader = SweepLoader(dir_manager,num_para_read=4)

output_files = sweep_reader.read_all_output_file_keys()
sweep_variables = sweep_reader.read_all_sweep_var_files()

print('Output files in json keys:')
pprint(sweep_reader.read_all_output_file_keys())
print()

print('Parameter sweep variables found:')
pprint(sweep_reader.read_all_sweep_var_files())
print()

#%%
# Now we use our sweep reader to read in the existing sweep results. We can also
# use a read configuration here if we want to filter out specific parts of the
# simulation output but for now we will read all of the data.

if __name__ == '__main__':
    start_time = time.perf_counter()
    sweep_data = sweep_reader.read_results_para()
    read_time_para = time.perf_counter() - start_time

print("-"*80)
print(f'Number of simulations outputs read: {len(sweep_data):d}')
print(f'Read time parallel   = {read_time_para:.6f} seconds')
print("-"*80)
