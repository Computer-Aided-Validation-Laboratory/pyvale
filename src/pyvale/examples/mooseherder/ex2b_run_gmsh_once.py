# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Running Gmsh once
================================================================================

In this example we will run a gmsh script to generate a mesh file using the
GmshRunner class.
"""

#%%
#
import time
from pathlib import Path
import pyvale as pyv
from pyvale.mooseherder import GmshRunner


#%%
#
gmsh_path = Path.home() / 'gmsh/bin/gmsh'
gmsh_runner = GmshRunner(gmsh_path)

gmsh_input = pyv.DataSet.sim_case_gmsh_file_path(case_num=17)
gmsh_runner.set_input_file(gmsh_input)


#%%
#
start_time = time.perf_counter()
gmsh_runner.run(gmsh_input,parse_only=True)
run_time = time.perf_counter() - start_time

print()
print("-"*80)
print(f'Gmsh run time = {run_time :.3f} seconds')
print("-"*80)
print()


