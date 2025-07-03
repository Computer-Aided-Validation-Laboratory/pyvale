'''
================================================================================
pyvale: the python computer aided validation engine

License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
'''
import time
from pathlib import Path
from mooseherder.mooseconfig import MooseConfig
from mooseherder.mooserunner import MooseRunner

USER_DIR = Path.home()

def main() -> None:
    config = {'main_path': USER_DIR / 'moose',
            'app_path': USER_DIR / 'proteus',
            'app_name': 'proteus-opt'}

    moose_config = MooseConfig(config)
    moose_runner = MooseRunner(moose_config)

    moose_runner.set_run_opts(n_tasks = 1, n_threads = 6, redirect_out = False)

    input_file = Path('data/moose_mesh_test/mesh-tester-3D.i')

    start_time = time.perf_counter()
    moose_runner.run(input_file)
    run_time = time.perf_counter() - start_time

    print()
    print("="*80)
    print(f'MOOSE run time = {run_time:.3f} seconds')
    print("="*80)
    print()


if __name__ == '__main__':
    main()

