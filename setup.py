#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

from setuptools import  setup, Extension
from Cython.Build import cythonize
import numpy
import sys
from glob import glob
import pybind11


debug_mode = '--debug' in sys.argv
if debug_mode:
    sys.argv.remove('--debug')

# check if we are on windows
is_windows = sys.platform.startswith("win")
is_mac = sys.platform == "darwin"

if is_windows:
    openmp_flags = ['/openmp']
    cpp_std_flag = '/std:c++17'
    compile_flags = [cpp_std_flag] + openmp_flags
    if debug_mode:
        compile_flags += ['/Od', '/Zi']
    else:
        compile_flags += ['/O2']
    link_flags = []
elif is_mac:
    openmp_flags = ['-Xclang', '-fopenmp']
    openmp_link_flag = '-lomp'
    cpp_std_flag = '-std=c++17'
    compile_flags = [cpp_std_flag] + openmp_flags

    if debug_mode:
        compile_flags += ['-O0', '-g']
        link_flags = [openmp_link_flag, '-g']
    else:
        compile_flags += ['-O3']
        link_flags = [openmp_link_flag]

else:
    openmp_flags = ['-fopenmp']
    cpp_std_flag = '-std=c++17'
    compile_flags = [cpp_std_flag] + openmp_flags
    if debug_mode:
        compile_flags += ['-O0', '-g']
        link_flags = openmp_flags + ['-g']
    else:
        compile_flags += ['-O3']
        link_flags = openmp_flags


ext_cython = Extension(
    "pyvale.sensorsim.cython.rastercyth",
    ["src/pyvale/sensorsim/cython/rastercyth.py"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=openmp_flags,
    extra_link_args=openmp_flags,
)

ext_dic = Extension(
    'pyvale.dic.dic2dcpp',
    sorted(glob("src/pyvale/dic/cpp/dic*.cpp")),
    language="c++",
    include_dirs=[pybind11.get_include()],
    extra_compile_args=compile_flags,
    extra_link_args=link_flags,
)

ext_modules = cythonize([ext_cython], annotate=True) + [ext_dic]

setup(
    ext_modules=cythonize(ext_modules,
                          annotate=True),
)
