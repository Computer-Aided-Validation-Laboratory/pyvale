from setuptools import  setup, Extension
from Cython.Build import cythonize
import numpy
import sys
import os
from glob import glob
import pybind11
import urllib.request
import tarfile




debug_mode = '--debug' in sys.argv
if debug_mode:
    sys.argv.remove('--debug')  # Remove so setup() doesn't get confused

if sys.platform.startswith("win"):
    openmp_arg = '/openmp'
else:
    openmp_arg = '-fopenmp'

ext_cython = Extension(
        "pyvale.cython.rastercyth",
        ["src/pyvale/cython/rastercyth.py",],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-ffast-math",openmp_arg],
        extra_link_args=[openmp_arg],
    )

ext_dic = Extension(
    'pyvale.dic2dcpp',
    sorted(glob("src/pyvale/dic/cpp/dic*.cpp")),
    language="c++",
    include_dirs=[pybind11.get_include()],
    extra_compile_args=['-g', '-O0', '-fopenmp'] if debug_mode else ['-O3', '-fopenmp'],
    extra_link_args=['-fopenmp'] + (['-g'] if debug_mode else []),
)

ext_modules = cythonize([ext_cython], annotate=True) + [ext_dic]

setup(
    ext_modules=cythonize(ext_modules,
                          annotate=True),
)
