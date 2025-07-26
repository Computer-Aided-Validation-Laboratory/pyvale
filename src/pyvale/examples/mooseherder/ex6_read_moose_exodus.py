"""
==============================================================================
EXMAPLE: Run MOOSE once with mooseherder and read the exodus output

Author: Lloyd Fletcher
==============================================================================
"""
import time
from pprint import pprint
from pathlib import Path
from typing import Any
import numpy as np
from mooseherder import (MooseRunner,
                         MooseConfig,
                         ExodusReader)

USER_DIR = Path.home()


def print_attrs(in_obj: Any) -> None:
    _ = [print(aa) for aa in dir(in_obj) if '__' not in aa]


def main() -> None:
    """main: run moose once and read the exodus output
    """
    print('='*80)
    print('EXAMPLE: Run MOOSE once, read exodus.')
    print('='*80)
    print('Generating exodus output to read by running MOOSE once.')

    # Create the moose runner with correct paths to moose and apps
    moose_config = MooseConfig().read_config(Path.cwd() / 'moose-config.json')
    moose_runner = MooseRunner(moose_config)

    # Set input and parallelisation options
    moose_runner.set_run_opts(n_tasks = 1, n_threads = 4,redirect_out = True)
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
    print('='*80)
    print()

    output_file = Path('scripts/moose/moose-mech-simple_out.e')
    print('Reading exodus file with ExodusReader:')
    print(output_file)
    print()

    exodus_reader = ExodusReader(output_file)

    print('Read all data in the exodus file returning a SimData object with everything.')
    all_sim_data = exodus_reader.read_all_sim_data()
    pprint(type(all_sim_data))
    print()
    print('Attributes of SimData:')
    print_attrs(all_sim_data)
    print()

    print('Read only specific node/element/global variables using SimReadConfig.')
    print('Use ExodusReader to get SimReadConfig with all variables then edit it.')
    read_config = exodus_reader.get_read_config()
    pprint(type(read_config))
    print()
    print('Attributes of SimReadConfig:')
    print_attrs(read_config)
    print()

    print('Set any attribute of SimReadConfig to None to not read data.')
    print('This example sets elem_vars to None and reads the exodus:')
    read_config.elem_vars = None
    sim_data = exodus_reader.read_sim_data(read_config)
    print(f'sim_data.elem_vars = {sim_data.elem_vars}')
    print()

    print('The attributes of SimReadConfig for time, coords and connect ')
    print('can be set to None or False to not be read or True to be read.')
    print('Here we turn off reading for time, coords and connect:')
    read_config.time = False
    read_config.coords = False
    read_config.connect = False
    sim_data = exodus_reader.read_sim_data(read_config)
    print(f'sim_data.time = {sim_data.time}')
    print(f'sim_data.coords = {sim_data.coords}')
    print(f'sim_data.connect = {sim_data.connect}')
    print()
    print('Turning back on reading for time, coords and connect.')
    read_config.time = True
    read_config.coords = True
    read_config.connect = True
    sim_data = exodus_reader.read_sim_data(read_config)
    print(f'sim_data.time = {sim_data.time}')
    print(f'sim_data.coords = {sim_data.coords}')
    print(f'sim_data.connect = {sim_data.connect}')
    print()

    print('Set attributes of SimReadConfig to a list of names as np.array')
    print('This example only reads disp_x from the nodal variables:')
    read_config.node_vars = np.array(['disp_x'])
    sim_data = exodus_reader.read_sim_data(read_config)
    print('sim_data.node_vars =')
    pprint(sim_data.node_vars)
    print()

    print('Element variables are an exception as they also need a block.')
    read_config.elem_vars = [('strain_xx',1),('strain_yy',1)]
    sim_data = exodus_reader.read_sim_data(read_config)
    print("sim_data.elem_vars[('strain_xx',1)] = ")
    pprint(sim_data.elem_vars[('strain_xx',1)]) # type: ignore
    print()

    print('NOTE: depending on output settings element variables may appear as nodal.')
    print('-'*80)
    print()


if __name__ == '__main__':
    main()
