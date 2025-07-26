'''
==============================================================================
EXAMPLE: Run MOOSE in sequential then parallel mode with mooseherder

Author: Lloyd Fletcher, Rory Spencer
==============================================================================
'''
from pathlib import Path
from mooseherder import (MooseHerd,
                         MooseRunner,
                         InputModifier,
                         DirectoryManager,
                         MooseConfig)

USER_DIR = Path.home()

def main():
    """main: run moose once, sequential then parallel.
    """
    print("-"*80)
    print('EXAMPLE: Herd Setup')
    print("-"*80)

    config_path = Path.cwd() / 'moose-config.json'
    moose_config = MooseConfig().read_config(config_path)
    moose_input = Path('scripts/moose/moose-mech-simple.i')

    moose_modifier = InputModifier(moose_input,'#','')
    moose_runner = MooseRunner(moose_config)
    moose_runner.set_run_opts(n_tasks = 1,
                          n_threads = 2,
                          redirect_out = True)

    dir_manager = DirectoryManager(n_dirs=4)

    # Start the herd and create working directories
    herd = MooseHerd([moose_runner],[moose_modifier],dir_manager)

    # Set the parallelisation options, we have 8 combinations of variables and
    # 4 MOOSE intances running, so 2 runs will be saved in each working directory
    herd.set_num_para_sims(n_para=4)

     # Send all the output to the examples directory and clear out old output
    dir_manager.set_base_dir(Path('examples/'))
    dir_manager.clear_dirs()
    dir_manager.create_dirs()

    # Create variables to sweep in a list of dictionaries, 8 combinations possible.
    n_elem_y = [10,20]
    e_mod = [1e9,2e9]
    p_rat = [0.3,0.35]
    moose_vars = list([])
    for nn in n_elem_y:
        for ee in e_mod:
            for pp in p_rat:
                # Needs to be list[list[dict]] - outer list is simulation iteration,
                # inner list is what is passed to each runner/inputmodifier
                moose_vars.append([{'n_elem_y':nn,'e_modulus':ee,'p_ratio':pp}])

    print('Herd sweep variables:')
    for vv in moose_vars:
        print(vv)

    print()
    print("-"*80)
    print('EXAMPLE: Run MOOSE once')
    print("-"*80)

    # Single run saved in sim-workdir-1
    herd.run_once(0,moose_vars[0])

    print(f'Run time (once) = {herd.get_iter_time():.3f} seconds')
    print("-"*80)
    print()

    print("-"*80)
    print('EXAMPLE: Run MOOSE sequentially')
    print("-"*80)

    # Run all variable combinations (8) sequentially in sim-workdir-1
    herd.run_sequential(moose_vars)

    print(f'Run time (seq) = {herd.get_sweep_time():.3f} seconds')
    print("-"*80)
    print()

    print("-"*80)
    print('EXAMPLE: Run MOOSE in parallel')
    print("-"*80)

    # Run all variable combinations across 4 MOOSE instances with two runs saved in
    # each sim-workdir
    herd.run_para(moose_vars)

    print(f'Run time (para) = {herd.get_sweep_time():.3f} seconds')
    print("-"*80)
    print()


if __name__ == '__main__':
    main()


