'''
================================================================================
Analytic test case data - linear

pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
'''
from pathlib import Path
import pyvale.mooseherder as mh

def main() -> None:
    moose_input = Path('simcases/case13.i')
    moose_modifier = mh.InputModifier(moose_input,'#','')

    moose_config = mh.MooseConfig({'main_path': Path.home() / 'moose',
                                   'app_path': Path.home() / 'proteus',
                                   'app_name': 'proteus-opt'})
    moose_runner = mh.MooseRunner(moose_config)
    moose_runner.set_run_opts(n_tasks = 1,
                              n_threads = 2,
                              redirect_out = False)

    dir_manager = mh.DirectoryManager(n_dirs=4)
    dir_manager.set_base_dir(Path('dev/lfdev/simdata/'))
    dir_manager.clear_dirs()
    dir_manager.create_dirs()

    herd = mh.MooseHerd([moose_runner],
                        [moose_modifier],
                        dir_manager)
    herd.set_num_para_sims(n_para=4)


    # Create variables to sweep in a list of dictionaries for mesh parameters
    # 2^3=8 combinations possible
    leng_x = 100e-3
    surf_heat_flux = 0.5e6
    therm_cond = 384.0
    sweep_limits = {'lengX': (0.9*leng_x,leng_x,1.1*leng_x),
                    'surfHeatFlux': (0.9*surf_heat_flux,surf_heat_flux,1.1*surf_heat_flux),
                    'cuThermCond': (0.9*therm_cond,therm_cond,1.1*therm_cond)}

    sweep_combinations = []
    for ll in sweep_limits['lengX']:
        for ss in sweep_limits['surfHeatFlux']:
            for tt in sweep_limits['cuThermCond']:
                sweep_combinations.append([])








if __name__ == "__main__":
    main()