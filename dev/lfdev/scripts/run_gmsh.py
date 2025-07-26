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
DATA_DIR = Path('check/')

def main() -> None:

    gmsh_path = USER_DIR / 'gmsh/bin/gmsh'
    gmsh_runner = GmshRunner(gmsh_path)

    gmsh_input = DATA_DIR / 'case11.geo'
    gmsh_runner.set_input_file(gmsh_input)

    print('Running gmsh...')
    print()

    gmsh_start = time.perf_counter()
    gmsh_runner.run(None,False)
    gmsh_run_time = time.perf_counter()-gmsh_start

    print()
    print("="*80)
    print(f'Gmsh run time = {gmsh_run_time:.2f} seconds')
    print("="*80)
    print()


if __name__ == '__main__':
    main()

