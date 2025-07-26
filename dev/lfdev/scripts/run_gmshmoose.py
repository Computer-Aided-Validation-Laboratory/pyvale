'''
================================================================================
pyvale: the python computer aided validation engine

License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
'''
import time
from pathlib import Path
from pyvale.mooseherder import (MooseConfig,
                        MooseRunner,
                        GmshRunner)

USER_DIR = Path.home()
DATA_DIR = Path('data/thermal_with_gmsh/')

def main() -> None:
    gmsh_runner = GmshRunner(USER_DIR / 'gmsh/bin/gmsh')

    gmsh_input = DATA_DIR / 'monoblock_3d.geo'
    gmsh_runner.set_input_file(gmsh_input)

    gmsh_start = time.perf_counter()
    gmsh_runner.run()
    gmsh_run_time = time.perf_counter()-gmsh_start


    config = {'main_path': USER_DIR / 'moose',
            'app_path': USER_DIR / 'proteus',
            'app_name': 'proteus-opt'}

    moose_config = MooseConfig(config)
    moose_runner = MooseRunner(moose_config)

    moose_runner.set_run_opts(n_tasks = 1, n_threads = 7, redirect_out = False)

    input_file = DATA_DIR / 'monoblock_gmsh_thermal.i'

    moose_start_time = time.perf_counter()
    moose_runner.run(input_file)
    moose_run_time = time.perf_counter() - moose_start_time

    print()
    print("="*80)
    print(f'Gmsh run time = {gmsh_run_time:.2f} seconds')
    print(f'MOOSE run time = {moose_run_time:.3f} seconds')
    print("="*80)
    print()


if __name__ == '__main__':
    main()

