# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import sys
from glob import glob
from setuptools import setup, Extension

debug_mode = '--debug' in sys.argv
if debug_mode:
    sys.argv.remove('--debug')  # Remove so setup() doesn't get confused


ext = Extension(
    'dic2dcpp',
    sorted(glob("cpp/dic*.cpp")),
    language="c++",
    extra_compile_args=['-g', '-O0', '-fopenmp'] if debug_mode else ['-O3', '-fopenmp'],
    extra_link_args=['-fopenmp'] + (['-g'] if debug_mode else []),
)

setup(
    name='dic2dcpp',
    author='Joel Hirst',
    version='0.1',
    ext_modules=[ext],
    zip_safe=False
)

