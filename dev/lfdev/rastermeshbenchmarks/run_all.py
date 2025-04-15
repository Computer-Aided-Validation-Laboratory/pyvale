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


USER_DIR = Path.home()
FORCE_GMSH = True
SIM_DIR = Path("dev/lfdev/rastermeshbenchmarks/")
SIM_STRS = ("case21_m1",
            "case21_m2",
            "case21_m3",
            "case21_m4",
            "case21_m5")

def run_one_case(sim_str: str) -> None:
    print(80*'=')
    print(f'Running: {sim_str}')
    print(80*'=')

    sim_files = (sim_str+'.geo',sim_str+'.i')

    # NOTE: if the msh file exists then gmsh will not run
    if (((SIM_DIR / sim_files[0]).is_file() and not
        (SIM_DIR / sim_files[0]).with_suffix('.msh').is_file()) or
        FORCE_GMSH):
        gmsh_runner = GmshRunner(USER_DIR / 'gmsh/bin/gmsh')

        gmsh_start = time.perf_counter()
        gmsh_runner.run(SIM_DIR / sim_files[0])
        gmsh_run_time = time.perf_counter()-gmsh_start
    else:
        print('Bypassing gmsh.')
        gmsh_run_time = 0.0

    config = {'main_path': USER_DIR / 'moose',
            'app_path': USER_DIR / 'proteus',
            'app_name': 'proteus-opt'}

    moose_config = MooseConfig(config)
    moose_runner = MooseRunner(moose_config)

    moose_runner.set_run_opts(n_tasks = 1,
                              n_threads = 8,
                              redirect_out = False)

    moose_start_time = time.perf_counter()
    moose_runner.run(SIM_DIR / sim_files[1])
    moose_run_time = time.perf_counter() - moose_start_time

    print()
    print("="*80)
    print(f'CASE: {sim_str}')
    print(f'Gmsh run time = {gmsh_run_time:.2f} seconds')
    print(f'MOOSE run time = {moose_run_time:.3f} seconds')
    print("="*80)
    print()


def main() -> None:
    for ss in SIM_STRS:
        run_one_case(ss)

if __name__ == '__main__':
    main()

