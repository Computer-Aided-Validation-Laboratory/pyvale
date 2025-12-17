#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

import subprocess
from pathlib import Path

def main() -> None:

    project_root = Path(__file__).resolve().parents[1]
    examples_basic = project_root/"src"/"pyvale"/"examples"/"extsensorsim"
    all_files = list(examples_basic.glob("*.py")) 

    example_files = list([])
    for ff in all_files:
        file_name = ff.name
        if "ex" in file_name:
            example_files.append(ff)

    example_files.sort()
    print('Running all examples files listed below:')
    for ff in example_files:
        print(ff)
    print()
    
    for ee in example_files:
        run_str = 'python '+ str(ee)
        print(run_str)
        subprocess.run(run_str, shell=True)

if __name__ == "__main__":
    main()
