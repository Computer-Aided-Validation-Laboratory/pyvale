'''
================================================================================
pyvale: the python computer aided validation engine

License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
'''

import os
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

    moose_runner.set_run_opts(n_tasks = 1, n_threads = 4, redirect_out = False)

    base_path = Path('data/')
    all_files = os.listdir(base_path)
    input_files = list([])
    for ff in all_files:
        if '.i' in ff:
            input_files.append(base_path / ff)

    for ii in input_files:
        moose_runner.run(ii)

if __name__ == '__main__':
    main()


