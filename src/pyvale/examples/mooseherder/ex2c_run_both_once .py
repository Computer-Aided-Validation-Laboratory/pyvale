"""
==============================================================================
EXAMPLE: Run gmsh+MOOSE once with mooseherder with MOOSE reading the gmsh
generated mesh.

Author: Lloyd Fletcher
==============================================================================
"""
import time
import os
import shutil
from pathlib import Path
from mooseherder import (MooseConfig,
                        GmshRunner,
                        MooseRunner)

USER_DIR = Path.home()

def main():
    """main: run gmsh and moose once
    """
    print("-"*80)
    print('EXAMPLE: Run Gmsh+MOOSE once')
    print("-"*80)

    gmsh_path = USER_DIR / 'gmsh/bin/gmsh'
    gmsh_runner = GmshRunner(gmsh_path)

    gmsh_input = Path('scripts/gmsh/gmsh_tens_spline_2d.geo')
    gmsh_runner.set_input_file(gmsh_input)

    print('Gmsh path:' + str(gmsh_path))
    print('Gmsh input:' + str(gmsh_input))
    print()

    print('Running gmsh...')
    print()

    gmsh_start = time.perf_counter()
    gmsh_runner.run()
    gmsh_run_time = time.perf_counter()-gmsh_start

    config_path = Path.cwd() / 'moose-config.json'

    print(f'Reading MOOSE config from: \n{str(config_path)}\n')

    moose_config = MooseConfig()
    moose_config.read_config(config_path)

    print('Creating the MooseRunner with the specified config.\n')

    moose_runner = MooseRunner(moose_config)


    print('Setting the input file and run parallelisation options.\n')

    moose_runner.set_run_opts(n_tasks = 1,
                          n_threads = 4,
                          redirect_out = True)
    input_file = Path('scripts/moose/moose-mech-gmsh.i')
    moose_runner.set_input_file(input_file)

    # Copy mesh file into the directory
    moose_mesh_file = os.path.split(input_file)[0]+'/mesh_tens_spline_2d.msh'
    moose_mesh_file = input_file.parent / 'mesh_tens_spline_2d.msh'
    msh_file = Path('scripts/gmsh/mesh_tens_spline_2d.msh')
    shutil.copyfile(msh_file,moose_mesh_file)

    # Run the MOOSE!
    print('Running moose with:')
    print(moose_runner.get_arg_list())
    print()

    moose_start = time.perf_counter()
    moose_runner.run()
    moose_run_time = time.perf_counter() - moose_start

    print(f'Gmsh run time = {gmsh_run_time:.2f} seconds')
    print(f'MOOOSE run time = {moose_run_time:.2f} seconds')
    print("-"*80)
    print()

if __name__ == '__main__':
    main()


