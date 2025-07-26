# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import time
from pathlib import Path
from mooseherder import GmshRunner

USER_DIR = Path.home()

print("-"*80)
print('EXAMPLE: Run Gmsh 2D once')
print("-"*80)
gmsh_path = USER_DIR / 'gmsh/bin/gmsh'
gmsh_runner = GmshRunner(gmsh_path)

gmsh_input = Path('scripts/gmsh/gmsh_tens_spline_2d.geo')
gmsh_runner.set_input_file(gmsh_input)

print('Gmsh path:' + str(gmsh_path))
print('Gmsh input:' + str(gmsh_input))

print('Running gmsh...')
start_time = time.perf_counter()
gmsh_runner.run(gmsh_input,parse_only=True)
run_time = time.perf_counter() - start_time

print()
print(f'Gmsh 2D run time = {run_time :.3f} seconds')
print("-"*80)
print()

print("-"*80)
print('EXAMPLE: Run Gmsh 3D once')
print("-"*80)
gmsh_path = USER_DIR / 'gmsh/bin/gmsh'
gmsh_runner = GmshRunner(gmsh_path)

gmsh_input = Path('scripts/gmsh/gmsh_tens_spline_3d.geo')
gmsh_runner.set_input_file(gmsh_input)

print('Gmsh path:' + str(gmsh_path))
print('Gmsh input:' + str(gmsh_input))

print('Running gmsh...')
start_time = time.perf_counter()
gmsh_runner.run(gmsh_input,parse_only=True)
run_time = time.perf_counter() - start_time

print()
print(f'Gmsh 3D run time = {run_time :.3f} seconds')
print("-"*80)
print()
