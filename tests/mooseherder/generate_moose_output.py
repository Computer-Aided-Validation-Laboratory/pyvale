'''
==============================================================================
GENERATE OUTPUTS FOR TESTS

Author: Lloyd Fletcher
==============================================================================
'''
import time
import os
from pathlib import Path
from mooseherder import MooseRunner

USER_DIR = Path().home()

print('------------------------------------------')
print('GENERATE MOOSE HERDER TEST CASES')
print('------------------------------------------')

# Create the moose runner with correct paths to moose and apps
moose_dir = USER_DIR / 'moose'
moose_app_dir = USER_DIR / 'proteus'
moose_app_name = 'proteus-opt'
moose_runner = MooseRunner(moose_dir,moose_app_dir,moose_app_name)

# Set input and parallelisation options
moose_runner.set_run_opts(n_tasks=4,
                      n_threads=2,
                      redirect_out=True)

# These are all the test cases required
input_files = [ Path('scripts/moose-mech-simple.i'), \
                Path('scripts/moose-mech-sub-dom.i'), \
                Path('scripts/moose-mech-sub-dom-order2.i'), \
                Path('scripts/moose-mech-two-beams.i'), \
                Path('scripts/moose-mech-thermal-exp.i')]

for ii in input_files:
    start_time = time.perf_counter()

    moose_runner.set_input_file(ii)
    print('Running moose test case with:')
    print(moose_runner.get_arg_list())

    moose_runner.run()
    run_time = time.perf_counter() - start_time
    print('MOOSE run time = {run_time:.3f} seconds')
    print()

print('Test data generation complete.')
print()
