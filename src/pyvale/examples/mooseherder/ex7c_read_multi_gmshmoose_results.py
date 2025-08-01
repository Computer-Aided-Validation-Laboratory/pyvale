# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Read parameter sweep results for a Gmsh and MOOSE simulation
================================================================================

In this example we ...

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

import time
from pathlib import Path
from pprint import pprint
from pyvale.mooseherder import (MooseHerd,
                                MooseRunner,
                                GmshRunner,
                                MooseConfig,
                                InputModifier,
                                DirectoryManager,
                                SweepReader)

NUM_PARA_RUNS = 3


# Setup the MOOSE input modifier and runner
moose_input = Path('scripts/moose/moose-mech-simple.i')
moose_modifier = InputModifier(moose_input,'#','')

moose_config = MooseConfig().read_config(Path.cwd() / 'moose-config.json')
moose_runner = MooseRunner(moose_config)
moose_runner.set_run_opts(n_tasks = 1,
                            n_threads = 2,
                            redirect_out = True)

# Setup Gmsh
gmsh_input = Path('scripts/gmsh/gmsh_tens_spline_2d.geo')
gmsh_modifier = InputModifier(gmsh_input,'//',';')

gmsh_path = Path.home() / 'gmsh/bin/gmsh'
gmsh_runner = GmshRunner(gmsh_path)
gmsh_runner.set_input_file(gmsh_input)

# Setup herd composition
sim_runners = [gmsh_runner,moose_runner]
input_modifiers = [gmsh_modifier,moose_modifier]
dir_manager = DirectoryManager(n_dirs=4)

# Start the herd and create working directories
herd = MooseHerd(sim_runners,input_modifiers,dir_manager)
# Set the parallelisation options, we have 8 combinations of variables and
# 4 MOOSE intances running, so 2 runs will be saved in each working directory
herd.set_num_para_sims(n_para=4)

    # Send all the output to the examples directory and clear out old output
dir_manager.set_base_dir(Path('examples/'))
dir_manager.clear_dirs()
dir_manager.create_dirs()

# Create variables to sweep in a list of dictionaries, 8 combinations possible.
p0 = [1E-3,2E-3]
p1 = [1.5E-3,2E-3]
p2 = [1E-3,3E-3]
var_sweep = list([])
for nn in p0:
    for ee in p1:
        for pp in p2:
            var_sweep.append([{'p0':nn,'p1':ee,'p2':pp},None])

print('Herd sweep variables:')
pprint(var_sweep)
print()

# Run all variable combinations across 4 MOOSE instances with two runs saved in
# each sim-workdir
if __name__ ==  "__main__":
    for rr in range(NUM_PARA_RUNS):
        herd.run_para(var_sweep)

        print(f'Run time (para {rr+1}) = {herd.get_sweep_time():.3f} seconds')
        print('------------------------------------------')

print("-"*80)
print('EXAMPLE: Read Herd Sweep Output')
print("-"*80)
sweep_reader = SweepReader(dir_manager,num_para_read=4)
output_files = sweep_reader.read_all_output_keys()

print('Herd output files (from output_keys.json):')
pprint(output_files)
print()

print("-"*80)
print('Reading all output files in parallel as list(SimData).')
print()

if __name__ ==  "__main__":
    start_time = time.perf_counter()
    read_all = sweep_reader.read_results_para()
    read_time_para = time.perf_counter() - start_time

print(f'Number of simulations outputs: {len(read_all):d}')
print()
print("="*80)
print(f'Read time parallel   = {read_time_para:.6f} seconds')
print("="*80)
print()

