'''
================================================================================
pyvale: the python computer aided validation engine

License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
'''
from pathlib import Path
import numpy as np
from mooseherder import (MooseHerd,
                         MooseRunner,
                         MooseConfig,
                         InputModifier,
                         DirectoryManager,
                         SweepReader)

USER_DIR = Path.home()
DATA_DIR = Path('dev/lf-dev/')

def main() -> None:
    moose_input = DATA_DIR / 'plate2d_therm_steady.i'
    moose_modifier = InputModifier(moose_input,'#','')

    config = {'main_path': USER_DIR / 'moose',
            'app_path': USER_DIR / 'proteus',
            'app_name': 'proteus-opt'}
    moose_config = MooseConfig(config)
    moose_runner = MooseRunner(moose_config)
    moose_runner.set_run_opts(n_tasks = 1,
                              n_threads = 4,
                              redirect_out = True)

    dir_manager = DirectoryManager(n_dirs=4)
    herd = MooseHerd([moose_runner],[moose_modifier],dir_manager)
    herd.set_num_para_sims(n_para=2)

    dir_manager.set_base_dir(DATA_DIR / 'sweep_data/')
    dir_manager.clear_dirs()
    dir_manager.create_dirs()

    xv1 = np.linspace(xlim1[0],xlim1[1],n)
    xv2 = np.linspace(xlim2[0],xlim2[1],n)
    xv = (xv1,xv2)
    (xm1,xm2) = np.meshgrid(*xv)

    x1_str = ''
    x2_str = ''
    x1 = (10,20)
    x2 = (1e9,2e9)
    moose_vars = list([])
    for xx1 in x1:
        for xx2 in x2:
                moose_vars.append([{x1_str:xx1,x2_str:xx2}])

    print(moose_modifier.get_vars())


if __name__ == "__main__":
    main()