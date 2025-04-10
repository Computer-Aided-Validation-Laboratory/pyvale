'''
================================================================================
pyvale: the python computer aided validation engine

License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
'''
import time
from pathlib import Path
from mooseherder import (MooseConfig,
                         MooseRunner,
                         GmshRunner)

#======================================
# Change this to run a different case
CASE_STR = 'case16'
#======================================

CASE_FILES = (CASE_STR+'.geo',CASE_STR+'.i')
CASE_DIR = Path('src/simcases/'+CASE_STR+'/')

USER_DIR = Path.home()

FORCE_GMSH = False

def main() -> None:
    # NOTE: if the msh file exists then gmsh will not run
    if (((CASE_DIR / CASE_FILES[0]).is_file() and not
        (CASE_DIR / CASE_FILES[0]).with_suffix('.msh').is_file()) or
        FORCE_GMSH):
        gmsh_runner = GmshRunner(USER_DIR / 'gmsh/bin/gmsh')

        gmsh_start = time.perf_counter()
        gmsh_runner.run(CASE_DIR / CASE_FILES[0])
        gmsh_run_time = time.perf_counter()-gmsh_start
    else:
        print('Bypassing gmsh.')
        gmsh_run_time = 0.0

    config = {'main_path': USER_DIR / 'moose',
            'app_path': USER_DIR / 'proteus',
            'app_name': 'proteus-opt'}

    print(config['main_path'])

    moose_config = MooseConfig(config)
    moose_runner = MooseRunner(moose_config)

    moose_runner.set_run_opts(n_tasks = 3,
                              n_threads = 2,
                              redirect_out = False)

    moose_start_time = time.perf_counter()
    moose_runner.run(CASE_DIR / CASE_FILES[1])
    moose_run_time = time.perf_counter() - moose_start_time

    print()
    print("="*80)
    print(f'Gmsh run time = {gmsh_run_time:.2f} seconds')
    print(f'MOOSE run time = {moose_run_time:.3f} seconds')
    print("="*80)
    print()

if __name__ == '__main__':
    main()

