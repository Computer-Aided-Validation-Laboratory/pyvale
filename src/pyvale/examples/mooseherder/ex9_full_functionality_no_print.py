# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Full end-to-end parameter sweep demonstration
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

from pathlib import Path
from pyvale.mooseherder import (MooseHerd,
                                MooseRunner,
                                MooseConfig,
                                InputModifier,
                                DirectoryManager,
                                SweepReader)

NUM_PARA_RUNS = 3

moose_input = Path('scripts/moose/moose-mech-simple.i')
moose_modifier = InputModifier(moose_input,'#','')

moose_config = MooseConfig().read_config(Path.cwd() / 'moose-config.json')
moose_runner = MooseRunner(moose_config)
moose_runner.set_run_opts(n_tasks = 1,
                            n_threads = 2,
                            redirect_out = False)

dir_manager = DirectoryManager(n_dirs=4)

herd = MooseHerd([moose_runner],[moose_modifier],dir_manager)
herd.set_num_para_sims(n_para=4)
herd.set_keep_flag(False)

dir_manager.set_base_dir(Path('examples/'))
dir_manager.clear_dirs()
dir_manager.create_dirs()

n_elem_y = [10,20]
e_mod = [1e9,2e9]
p_rat = [0.3,0.35]
moose_vars = list([])
for nn in n_elem_y:
    for ee in e_mod:
        for pp in p_rat:
            moose_vars.append([{'n_elem_y':nn,'e_modulus':ee,'p_ratio':pp}])


if __name__ == '__main__':
    for _ in range(NUM_PARA_RUNS):
        herd.run_para(moose_vars)

    sweep_reader = SweepReader(dir_manager,num_para_read=4)
    sweep_reader.read_all_output_keys()
    read_all = sweep_reader.read_results_para()

