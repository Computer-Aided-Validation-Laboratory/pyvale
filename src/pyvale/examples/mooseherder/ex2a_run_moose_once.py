"""
==============================================================================
EXAMPLE: Run MOOSE using mooseherder once

Author: Lloyd Fletcher
==============================================================================
"""
import time
from pathlib import Path
from mooseherder import (MooseConfig,
                         MooseRunner)

def main() -> None:
    """main: run moose once with runner class
    """
    print("-"*80)
    print('EXAMPLE: Run MOOSE once')
    print("-"*80)

    config_path = Path.cwd() / 'moose-config.json'
    moose_config = MooseConfig().read_config(config_path)
    print(f'Reading MOOSE config from: \n{str(config_path)}\n')

    print('Creating the MooseRunner with the specified config.\n')
    moose_runner = MooseRunner(moose_config)

    print('Setting the input file and run parallelisation options.\n')

    moose_runner.set_run_opts(n_tasks = 1,
                              n_threads = 8,
                              redirect_out = False)

    input_file = Path('scripts/moose/moose-mech-simple.i')
    moose_runner.set_input_file(input_file)

    # Run the MOOSE!
    print('Running moose with:')
    print(moose_runner.get_arg_list())

    start_time = time.perf_counter()
    moose_runner.run()
    run_time = time.perf_counter() - start_time

    print()
    print(f'MOOSE run time = {run_time:.3f} seconds')
    print("-"*80)
    print()

if __name__ == '__main__':
    main()

