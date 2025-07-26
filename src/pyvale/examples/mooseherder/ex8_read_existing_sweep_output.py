'''
==============================================================================
EXAMPLE: Read existing sweep results

Author: Lloyd Fletcher
==============================================================================
'''
import time
from pprint import pprint
from pathlib import Path
from mooseherder import DirectoryManager
from mooseherder import SweepReader


def main() -> None:
    """main:
    """
    print("-"*80)
    print('EXAMPLE: Read Existing Sweep Results')
    print("-"*80)

    base_dir = Path('examples/example_output/')
    print(f'Reading from sub directories in:\n{base_dir}\n')

    dir_manager = DirectoryManager(n_dirs=4)
    dir_manager.set_base_dir(base_dir)

    sweep_reader = SweepReader(dir_manager,num_para_read=4)
    output_files = sweep_reader.read_all_output_keys()
    sweep_variables = sweep_reader.read_all_sweep_var_files()

    print('Output files in json keys:')
    pprint(output_files)
    print()

    print('Sweep variables:')
    pprint(sweep_variables)
    print()

    print("-"*80)
    print('Reading all output files in parallel as list(SimData).')
    start_time = time.perf_counter()
    sweep_data = sweep_reader.read_results_para()
    read_time_para = time.perf_counter() - start_time

    print(f'Number of simulations outputs read: {len(sweep_data):d}')
    print()
    print("="*80)
    print(f'Read time parallel   = {read_time_para:.6f} seconds')
    print("="*80)


if __name__ == '__main__':
    main()
