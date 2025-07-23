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
if is_windows:
    openmp_flag = '/openmp'
    cpp_std_flag = '/std:c++17'
    compile_flags = [cpp_std_flag, openmp_flag]
    if debug_mode:
        compile_flags += ['/Od', '/Zi']
    else:
        compile_flags += ['/O2']
    link_flags = []
else:
    openmp_flag = '-fopenmp'
    cpp_std_flag = '-std=c++17'
    compile_flags = [cpp_std_flag, openmp_flag]
    if debug_mode:
        compile_flags += ['-O0', '-g']
        link_flags = [openmp_flag, '-g']
    else:
        compile_flags += ['-O3']
        link_flags = [openmp_flag]


ext_cython = Extension(
    "pyvale.cython.rastercyth",
    ["src/pyvale/cython/rastercyth.py"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=[openmp_flag],
    extra_link_args=[openmp_flag],
)



ext_dic = Extension(
    'pyvale.dic2dcpp',
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
