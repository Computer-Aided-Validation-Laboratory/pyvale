'''
==============================================================================
RUN ALL: helper script to run all examples to check for errors

Author: Lloyd Fletcher
==============================================================================
'''
import os
import subprocess
from pathlib import Path
from pprint import pprint

EXAMPLE_DIR = Path('examples/')
all_files = os.listdir(EXAMPLE_DIR)

example_files = list([])
for ff in all_files:
    if ('ex' in ff) and ('.py' in ff):
        example_files.append(EXAMPLE_DIR / ff)

example_files.sort()
print('Running all examples files listed below:')
pprint(example_files)
print()

for ee in example_files:
    run_str = 'python '+ str(ee)
    print(run_str)
    subprocess.run(run_str, shell=True)

