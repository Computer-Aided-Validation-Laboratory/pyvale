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


NUM_CASES = 23
USER_DIR = Path.home()
FORCE_GMSH = False
CASES_DIR = Path('src/simcases/')

def run_one_case(case_str: str) -> None:
    print(80*'=')
    print(f'Running: {case_str}')
    print(80*'=')

    case_files = (case_str+'.geo',case_str+'.i')
    case_dir =  CASES_DIR / case_str

    gmsh_run_time = 0.0
    if (case_dir / case_files[0]).is_file():
        gmsh_runner = GmshRunner(USER_DIR / 'gmsh/bin/gmsh')

        gmsh_start = time.perf_counter()
        gmsh_runner.run(case_dir / case_files[0])
        gmsh_run_time = time.perf_counter()-gmsh_start

    config = {'main_path': USER_DIR / 'moose',
            'app_path': USER_DIR / 'proteus',
            'app_name': 'proteus-opt'}

    moose_config = MooseConfig(config)
    moose_runner = MooseRunner(moose_config)

    moose_runner.set_run_opts(n_tasks = 1,
                              n_threads = 8,
                              redirect_out = False)

    moose_start_time = time.perf_counter()
    moose_runner.run(case_dir / case_files[1])
    moose_run_time = time.perf_counter() - moose_start_time

    print()
    print("="*80)
    print(f'CASE: {case_str}')
    print(f'Gmsh run time = {gmsh_run_time:.2f} seconds')
    print(f'MOOSE run time = {moose_run_time:.3f} seconds')
    print("="*80)
    print()


def main() -> None:
    for ss in range(NUM_CASES):
        case_str = 'case' + str(ss+1).zfill(2)
        run_one_case(case_str)


if __name__ == '__main__':
    main()

