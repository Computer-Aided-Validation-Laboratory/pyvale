from setuptools import  setup, Extension
from Cython.Build import cythonize
import numpy
import sys

if sys.platform.startswith("win"):
    openmp_arg = '/openmp'
else:
    openmp_arg = '-fopenmp'

ext_modules = [
    Extension(
        "pyvale.cython.rastercyth",
        ["src/pyvale/cython/rastercyth.py",],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-ffast-math",openmp_arg],
        extra_link_args=[openmp_arg],
    ),
]

setup(
      ext_modules=cythonize(ext_modules, annotate=True),
)
